#coding=utf-8
import face_model
import argparse
import os,shutil
import numpy as np
import datetime
import util
import struct
import cv2
import requests
import redis

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='./facerec/models/fr/model-0000', help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int, help='gpu id')
args = parser.parse_args()

# 读入模型
model = face_model.FaceModel(args)

# 初始化所有信息
rootDir = './facerec/data'
tmpDir = rootDir + '/tmp'
registerDir = rootDir + '/register'
recognizeDir = rootDir + '/recognize'
if os.path.exists(rootDir) == False:
    os.mkdir(rootDir)
if os.path.exists(registerDir) == False:
    os.mkdir(registerDir)
if os.path.exists(recognizeDir) == False:
    os.mkdir(recognizeDir)
if os.path.exists(tmpDir) == False:
    os.mkdir(tmpDir)
allPersons = []  # 保存所有已注册人信息

def matchPerson(featsT, onePerson):
    '''
    获取与一个注册人脸匹配的最高分
    :param featsT:  待匹配人脸的特征
    :param onePerson:  其中一个注册人脸的信息
    :return:  返回与这个人脸匹配的最高分
    '''
    maxScore = 0
    featLen = len(onePerson['feat'])
    for idx in range(featLen):
        score = np.dot(onePerson['feat'][idx], featsT)
        if score > maxScore:
            maxScore = score
    return maxScore

def matchAllPerson(featsT, persons):
    '''
    当前人脸特征与所有注册人脸匹配，找出最佳匹配的人的Id，以及对应的分数
    :param featsT:  待识别人脸特征
    :param persons:  所有注册人脸信息
    :return:  返回最佳匹配人脸id以及最高分数值
    '''
    maxScore = -1
    bestId = 0
    for idx in range(len(persons)):
        score = matchPerson(featsT, persons[idx])
        if score > maxScore:
            maxScore = score
            bestId = idx
    return bestId, maxScore

def  getNormalFaces(img, json_dict):
    '''
    从上传图像中提取所有normalFace
    :param img:  上传的原始图片
    :param json_dict:  上传的原始json
    :return:  返回上传图片中的所有normalFace以及在上传图片中人脸框
    '''
    normalFaces = []
    bbox = []
    if 'faceList' not in json_dict.keys():
        return normalFaces,bbox
    son_dict = json_dict.get('faceList')
    for idx in range(len(son_dict)):
        if 'ldmark_pts' not in son_dict[idx].keys():
            continue
        dson_dict = son_dict[idx]
        rect = [dson_dict.get('x'), dson_dict.get('y'), dson_dict.get('w'), dson_dict.get('h')]
        ldmark = []
        for idx_ldmark in range(68):
            x = dson_dict.get('ldmark_pts')[idx_ldmark].get('x')
            y = dson_dict.get('ldmark_pts')[idx_ldmark].get('y')
            ldmark.append([x,y])
        normal = model.get_input_by_ldmark68(img,rect,ldmark)
        normalFaces.append(normal)
        bbox.append(rect)
    return normalFaces,bbox

def identifyFunc(img, json_dict, persons):
    '''
    识别接口
    :param normalFaces:  待识别的每个normalFace
    :param bbox:   待识别人脸框
    :param persons:   所有注册人脸信息
    :return: 识别结果
    '''
    normalFaces, boxes = getNormalFaces(img, json_dict)
    identifyInfo = []
    for idx in range(len(normalFaces)):
        # 人脸识别
        feats = model.get_feature(normalFaces[idx])
        featsT = feats.T
        bestId, maxScore = matchAllPerson(featsT, persons)
        info={}
        info['box'] = boxes[idx]
        info['userId'] = persons[bestId]['userId']
        info['score'] = maxScore
        identifyInfo.append(info)
    return identifyInfo

def registerFunc(img, json_dict):
    '''
    保存一个人的注册信息
    :param img:  上传的原始图片
    :param json_dict:  上传的原始json
    :param registerDir:  这个人保存的路径
    :return:  这个人需要保存的信息
    '''
    userId = ''
    if 'userId' in json_dict.keys():
        userId = json_dict.get('userId')
    info = {}
    normalFaces, boxes = getNormalFaces(img, json_dict)
    person_feat = []
    for idx in range(len(normalFaces)):
        feats = model.get_feature(normalFaces[idx])
        person_feat.append(feats)
    info['img'] = img
    info['json'] = json_dict
    info['userId'] = userId
    info['feat'] = person_feat
    return True,info

def fr_init():
    '''
    :return:  初始化操作，返回所有人注册信息
    '''
    global registerDir,allPersons
    allPersons = util.readAllPersonInfo()
    return allPersons

def  fr_run(imgStr, img, json_dict):
    '''
    :param img:   上传的一张图片，可以为一个人或者多个人
    :param json_dict:  上传图片对应的json
    :return:  处理结果
    '''
    global registerDir, recognizeDir,allPersons
    dstJson = json_dict
    if 'func' not in json_dict.keys():
        dstJson.update({'result': u'failure'})
        return dstJson
    func = json_dict.get('func')

    #cv2.imshow("test", img)
    #cv2.waitKey(0)

    # 根据功能选择不同的函数
    if func == 'register':   # 如果是注册，则保存这个人的信息，需要写如到数据库
        flag,person_info = registerFunc(img, json_dict)
        if flag == True:
            # 保存到数据库中
            ret = util.writeOnePersonInfo(imgStr, person_info)
            if ret == 'successful':
                fr_init()
                dstJson.update({'result': u'successful'})
            else:
                dstJson.update({'result': u'failure'})
        else:
            dstJson.update({'result': u'failure'})
    elif func == 'idenfication':
        info = identifyFunc(img, json_dict, allPersons)
        dstJson = util.updateJson(dstJson, info)
        dstJson.update({'result': u'successful'})
        sonDir = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
        dstDir = os.path.join(recognizeDir, sonDir)
        util.saveSrcInfo(dstDir, img, dstJson)
    # 显示效果
    #util.drawInfo(img, dstJson)
    return dstJson

def  processAllPerson():
    global tmpDir
    for (path, dirnames, filenames) in os.walk(tmpDir):
        for filename in dirnames:
            personDir = os.path.join(tmpDir, filename)
            # 读入img和json，如果正确的时候才进入识别
            imgFlag, jsonFlag, img, json_dict = util.loadClientInfo(personDir)
            if imgFlag == True and jsonFlag == True:
                dstJson = fr_run('adf', img, json_dict)
            # 将处理完成后的任务删除
            #if os.path.exists(personDir):
            #    shutil.rmtree(personDir)
            #print(dstJson)

fr_init()

def  testWritePersonInfo():
    personDir = 'C:/Users/Administrator/Desktop/attence/ai-deploy/facerec/data/tmp/18696102288'
    encode_data = personDir + '/imgStr.dat'
    f = open(encode_data, 'rb')
    imgStr = f.read()
    imgFlag, jsonFlag, img, json_dict = util.loadClientInfo(personDir)
    dstJson = fr_run(imgStr, img, json_dict)

if __name__=='__main__':
    #测试处理tmp文件夹中的所有人
    #processAllPerson()
    testWritePersonInfo()
