#coding=utf-8
import cv2
import os
import numpy as np
import json
import base64
import requests
import redis

def  filterUserId(red, userId, allPerson):
    #if red.exists(userId) == False:  # 如果这个人不存在
    #    return False
    for idx in range(len(allPerson)):
        if userId == allPerson[idx]['userId']:  # 如果这个人已经读取过了
            return False
    return True

def  covByteToFeats(src):
    featStr = src[0].decode(encoding='utf-8')
    featSplit = featStr.split(',')
    featDst = []
    for idx in range(len(featSplit)):
        val = float(featSplit[idx])
        featDst.append(val)
    return featDst

def readOnePersonInfo(red, userId):
    '''
    从redis中读取一个人的注册信息
    :param red: redis对象
    :param userId: 这个人的ID
    :return: 这个人的注册信息，ID+feats
    '''
    strFeat0 = red.hmget(userId, 'feat_0')
    strFeat1 = red.hmget(userId, 'feat_1')
    strFeat2 = red.hmget(userId, 'feat_2')
    strFeat3 = red.hmget(userId, 'feat_3')
    strFeat4 = red.hmget(userId, 'feat_4')
    strFeat5 = red.hmget(userId, 'feat_5')
    feat0 = covByteToFeats(strFeat0)
    feat1 = covByteToFeats(strFeat1)
    feat2 = covByteToFeats(strFeat2)
    feat3 = covByteToFeats(strFeat3)
    feat4 = covByteToFeats(strFeat4)
    feat5 = covByteToFeats(strFeat5)
    userName = red.hmget(userId, 'realName')
    nameStr = userName[0].decode(encoding='utf-8')
    info = {}
    info['userId'] = userId
    info['userName'] = nameStr
    info['feat'] = [feat0, feat1,feat2, feat3,feat4,feat5]
    return info

def readAllPersonInfo():
    '''
    从redis中读取所有注册人脸信息
    :return: 所有人脸信息
    '''
    red = redis.Redis(host='192.168.50.163', port=6379, db=1)
    userIdList = red.lrange('jobNumberList', 0, -1)
    allPerson = []
    for idx in range(len(userIdList)):
        userId = userIdList[idx].decode(encoding='utf-8')
        if filterUserId(red, userId, allPerson) == False:
            continue
        info = readOnePersonInfo(red, userId)
        allPerson.append(info)
    return allPerson

def writeOnePersonInfo(imgStr, info):
    url = "http://192.168.50.163:8444/checking-in/clockInUser/saveClockInUser"
    userInfo = {}
    userInfo['userId'] = info['userId']   #string   用户工号
    userInfo['img'] = imgStr              #string 原始图像经过base64编码后的图像
    userInfo['json'] = json.dumps(info['json'])  #string 原始json
    userInfo['feat_0'] = info['feat'][0]
    userInfo['feat_1'] = info['feat'][1]
    userInfo['feat_2'] = info['feat'][2]
    userInfo['feat_3'] = info['feat'][3]
    userInfo['feat_4'] = info['feat'][4]
    userInfo['feat_5'] = info['feat'][5]
    retStr = requests.post(url, data=userInfo)
    retJson = json.loads(retStr.text)
    if retStr.status_code != 200:
        return 'failure'
    elif retJson['code'] == 1:
        return 'failure'
    return 'successful'

# 将client端所有信息load进来
def  loadClientInfo(clientDir):
    imgFlag = False
    jsonFlag = False
    imgFn = os.path.join(clientDir, 'info.jpg')
    img = cv2.imread(imgFn)
    try:
        imgFlag = True
    except:
        imgFlag = False
    jsonFn = os.path.join(clientDir, 'info.json')
    if os.path.exists(jsonFn) == True:
            jsonFlag = True
    with open(jsonFn, 'r') as load_f:
        json_dict = json.load(load_f)
    return imgFlag, jsonFlag, img, json_dict

def  iou(rect1, rect2):
    '''
    计算两个rect的iou
    :param rect1:
    :param rect2:
    :return: 返回这两个rect的重叠度
    '''
    x1 = int(rect1[0])
    y1 = int(rect1[1])
    width1 = int(rect1[2])
    height1 = int(rect1[3])

    x2 = int(rect2[0])
    y2 = int(rect2[1])
    width2 = int(rect2[2])
    height2 = int(rect2[3])

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0  # 重叠率为 0
    else:
        Area = width * height  # 两矩形相交面积
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    # return IOU
    return ratio

def  findRectId(srcJson, rect):
    '''
    获取当前rect在原始json中的索引
    :param srcJson:  原始json
    :param rect:  当前rect
    :return:  索引值
    '''
    son_dict = srcJson.get('faceList')
    for idx in range(len(son_dict)):
        x = son_dict[idx]['x']
        y = son_dict[idx]['y']
        w = son_dict[idx]['w']
        h = son_dict[idx]['h']
        json_rect = [x,y,w,h]
        ov = iou(rect, json_rect)
        if ov > 0.5:
            return idx
    return -1

def  updateJson(srcJson, info):
    dstJson = srcJson
    for idx in range(len(info)):
        rect = info[idx]['box']
        rect_idx = findRectId(srcJson, rect)
        if rect_idx < 0:
            continue
        userId = str(info[idx]['userId'])
        userName = str(info[idx]['userName'])
        score = float(info[idx]['score'])
        dstJson['faceList'][rect_idx].update({'userId': userId, 'score': score, 'userName': userName})
    return dstJson

def saveSrcInfo(saveDir, img, json_dict):
    if os.path.exists(saveDir) == False:
        os.mkdir(saveDir)
    imgFn = os.path.join(saveDir, 'info.jpg')
    cv2.imwrite(imgFn, img)
    jsonFn = os.path.join(saveDir, 'info.json')
    #json_str = json.dumps(json_dict,encoding='UTF-8',default=str)
    json_str = json.dumps(json_dict, default=str)
    with open(jsonFn, 'w') as json_file:
        json_file.write(json_str)

def drawInfo(img, json_dict):
    son_dict = json_dict.get('faceList')
    for idx in range(len(son_dict)):
        x = son_dict[idx]['x']
        y = son_dict[idx]['y']
        w = son_dict[idx]['w']
        h = son_dict[idx]['h']
        box = [x,y,w,h]
        userId = son_dict[idx]['userId']
        score = son_dict[idx]['score']
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(userId) + ',' + '%.2f' % (score)
        cv2.putText(img, text, (box[0], box[1] + 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('img', img)
    cv2.waitKey(0)

def drawRegisterInfo(img, json_dict):
    userName = json_dict['userName']
    userId = json_dict['userId']
    son_dict = json_dict.get('faceList')
    for idx in range(len(son_dict)):
        x = son_dict[idx]['x']
        y = son_dict[idx]['y']
        w = son_dict[idx]['w']
        h = son_dict[idx]['h']
        box = [x,y,w,h]
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 2)

        for idx_ldmark in range(68):
            x = son_dict[idx].get('ldmark_pts')[idx_ldmark].get('x')
            y = son_dict[idx].get('ldmark_pts')[idx_ldmark].get('y')
            cv2.circle(img, (x,y),2, (0,255,0))

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(userName) + ',' + str(userId)
        cv2.putText(img, text, (box[0], box[1] + 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('img', img)
    cv2.waitKey(0)

if __name__=='__main__':
    # dir = './facerec/data/recognize/20190507141601212000'
    #imgFlag, jsonFlag, img, json_dict = loadClientInfo(dir)
    #drawInfo(img, json_dict)

    dir = './facerec/data/register'
    '''
    imgFlag, jsonFlag, img, json_dict = loadClientInfo(dir)
    saveDir = './facerec/data/register'
    saveSrcInfo(saveDir, img, json_dict)
    '''

    #drawRegisterInfo(img, json_dict)
    #cv2.imshow('img', img)
    #cv2.waitKey(0)

   # 用python解析opencv 编码的base64
    encode_data = dir + '/base64_jpg.dat'
    f = open(encode_data, 'rb')
    f_read = f.read()
    #f_read_decode = f_read.decode('base64')
    f_read_decode = base64.b64decode(f_read)
    image = np.asarray(bytearray(f_read_decode), dtype="uint8")
    img_np = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imshow("test", img_np)
    cv2.waitKey(0)

