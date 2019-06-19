#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from skimage import transform as trans

def read_image(img_path, **kwargs):
  mode = kwargs.get('mode', 'rgb')
  layout = kwargs.get('layout', 'HWC')
  if mode=='gray':
    img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  else:
    img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR)
    if mode=='rgb':
      #print('to rgb')
      img = img[...,::-1]
    if layout=='CHW':
      img = np.transpose(img, (2,0,1))
  return img

def preprocess(img, bbox=None, landmark=None, **kwargs):
  if isinstance(img, str):
    img = read_image(img, **kwargs)
  M = None
  image_size = []
  str_image_size = kwargs.get('image_size', '')
  if len(str_image_size)>0:
    image_size = [int(x) for x in str_image_size.split(',')]
    if len(image_size)==1:
      image_size = [image_size[0], image_size[0]]
    assert len(image_size)==2
    assert image_size[0]==112
    assert image_size[0]==112 or image_size[1]==96
  if landmark is not None:
    assert len(image_size)==2
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
      src[:,0] += 8.0
    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    #M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

  if M is None:
    if bbox is None: #use center crop
      det = np.zeros(4, dtype=np.int32)
      det[0] = int(img.shape[1]*0.0625)
      det[1] = int(img.shape[0]*0.0625)
      det[2] = img.shape[1] - det[0]
      det[3] = img.shape[0] - det[1]
    else:
      det = bbox
    margin = kwargs.get('margin', 44)
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
    bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
    ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
    if len(image_size)>0:
      ret = cv2.resize(ret, (image_size[1], image_size[0]))
    return ret 
  else: #do align using landmark
    assert len(image_size)==2

    #src = src[0:3,:]
    #dst = dst[0:3,:]


    #print(src.shape, dst.shape)
    #print(src)
    #print(dst)
    #print(M)
    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)

    #tform3 = trans.ProjectiveTransform()
    #tform3.estimate(src, dst)
    #warped = trans.warp(img, tform3, output_shape=_shape)
    return warped

def cov_ldmark68_2_ldmark5(ldmark68):
    ldmark5 = []
    # 左眼
    left_eye_x = 0
    left_eye_y = 0
    for idx in range(36, 42):
      left_eye_x += ldmark68[idx][0]
      left_eye_y += ldmark68[idx][1]
    left_eye_x /= 6
    left_eye_y /= 6
    ldmark5.append([left_eye_x, left_eye_y])
    # 右眼
    right_eye_x = 0
    right_eye_y = 0
    for idx in range(42, 48):
      right_eye_x += ldmark68[idx][0]
      right_eye_y += ldmark68[idx][1]
    right_eye_x /= 6
    right_eye_y /= 6
    ldmark5.append([right_eye_x, right_eye_y])

    ldmark5.append(ldmark68[30])  # nose
    ldmark5.append(ldmark68[48])  # mouth_left
    ldmark5.append(ldmark68[54])  # mouth_right
    return ldmark5

def aligment_b5(image, faceRect, ldmark5):
  img_size = '112,112'
  box = [int(faceRect[0]), int(faceRect[1]), int(faceRect[0] + faceRect[2]), int(faceRect[1] + faceRect[3])]
  ldmark5_np = np.array([
    [ldmark5[0][0], ldmark5[0][1]],
    [ldmark5[1][0], ldmark5[1][1]],
    [ldmark5[2][0], ldmark5[2][1]],
    [ldmark5[3][0], ldmark5[3][1]],
    [ldmark5[4][0], ldmark5[4][1]], ], dtype=np.float32)
  warped = preprocess(image, bbox=box, landmark=ldmark5_np, image_size=img_size)
  '''
  cv2.imshow("warped", warped)
  cv2.rectangle(image, (faceRect[0], faceRect[1]), (faceRect[0] + faceRect[2], faceRect[1] + faceRect[3]),(0, 0, 255), 2)
  for pt in ldmark68:
      cv2.circle(image, (pt[0], pt[1]), 2, (255, 0, 0), -1)
  for pt in ldmark5:
      cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
  '''
  return warped

def aligment_by68(image, faceRect, ldmark68):
  ldmark5 = cov_ldmark68_2_ldmark5(ldmark68)
  # img_size = '%d,%d' % (frame.shape[0], frame.shape[1])
  img_size = '112,112'
  box = [int(faceRect[0]), int(faceRect[1]), int(faceRect[0] + faceRect[2]), int(faceRect[1] + faceRect[3])]
  ldmark5_np = np.array([
    [ldmark5[0][0], ldmark5[0][1]],
    [ldmark5[1][0], ldmark5[1][1]],
    [ldmark5[2][0], ldmark5[2][1]],
    [ldmark5[3][0], ldmark5[3][1]],
    [ldmark5[4][0], ldmark5[4][1]], ], dtype=np.float32)
  warped = preprocess(image, bbox=box, landmark=ldmark5_np, image_size=img_size)
  '''
  cv2.imshow("warped", warped)
  cv2.rectangle(image, (faceRect[0], faceRect[1]), (faceRect[0] + faceRect[2], faceRect[1] + faceRect[3]),(0, 0, 255), 2)
  for pt in ldmark68:
      cv2.circle(image, (pt[0], pt[1]), 2, (255, 0, 0), -1)
  for pt in ldmark5:
      cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
  '''
  return warped
