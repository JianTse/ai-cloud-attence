#!/usr/bin/python
# -*- coding: UTF-8 -*-
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import mxnet as mx
import cv2
import face_preprocess

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split('-')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])

  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
    def __init__(self, args):
        self.args = args
        if args.gpu < 0:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(args.gpu)
        _vec = args.image_size.split(',')
        assert len(_vec)==2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.model = None
        if len(args.model)>0:
            self.model = get_model(ctx, image_size, args.model, 'fc1')
        self.image_size = image_size

    def get_input_by_ldmark5(self, img, box, ldmark5):
        bbox = box
        points = ldmark5.reshape((2,5)).T
        #print(bbox)
        #print(points)
        nimg = face_preprocess.preprocess(img, bbox, points, image_size='112,112')
        #nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        #aligned = np.transpose(nimg, (2,0,1))
        return nimg

    def get_input_by_ldmark68(self, img, box, ldmark68):
        nimg = face_preprocess.aligment_by68(img, box, ldmark68)
        #nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        #aligned = np.transpose(nimg, (2,0,1))  # 换通道了，hwc --> chw
        #cv2.imshow('normal', nimg)
        #cv2.waitKey(0)
        return nimg

    def get_feature(self, img):
        #cv2.imshow('src_fr', img)
        #cv2.waitKey(1)
        img_dst = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # color transform:BGR---RGB
        img_dst = np.transpose(img_dst, (2, 0, 1))
        input_blob = np.expand_dims(img_dst, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()[0]
        l2_norm = cv2.norm(embedding, cv2.NORM_L2)
        return embedding / l2_norm

