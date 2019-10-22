#coding=utf-8
from flask import Flask,jsonify,request
from run import fr_run
import numpy as np
import cv2
import json
import base64

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        else:
            return super(NpEncoder, self).default(obj)
			
app = Flask(__name__)
@app.route('/ai_fr',methods=['post'])
def ai_fr():
    try:
        strJson = request.form['json']
        print ('srcJson: '+ strJson)
        src_json = json.loads(strJson)
        # 解析编码后的图像
        imgStr = request.form['image']
        #print ('srcImg: ' + imgStr)
        #f_read_decode = imgStr.decode('base64')
        f_read_decode = base64.b64decode(imgStr)
        image = np.asarray(bytearray(f_read_decode), dtype="uint8")
        cv_img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        #cv2.imshow('img', cv_img)
        #cv2.waitKey(0)

        # 云端计算结果，返回json
        dst_json = fr_run(imgStr,cv_img, src_json)
        #strJson = json.dumps(dst_json, ensure_ascii=False)
        strJson = json.dumps(dst_json, cls=MyEncoder)	
        print ('dstJson: ' + strJson)
        return strJson
    except Exception as ex:
        print(ex)
        json_dict = {}
        json_dict['result'] = 'failure'
        strJson = json.dumps(json_dict, ensure_ascii=False)
        print (strJson)
        return strJson

app.run(host='0.0.0.0',port=8802,debug=True)