import threading
import time
import json
import os
from flask import Flask, request, jsonify, send_file, url_for, Response
import numpy as np
import cv2
from image_engine import BingImageEngine
from swapper import *


app = Flask(__name__)

#swapper = Swapper()
#swapper.start_img_engine()
#swapper.start_fs_engine()

user_cache = 'user.json'
if not os.path.exists(user_cache):
    with open(user_cache,'w') as f:
        json.dump(dict(),f)

with open(user_cache,"r") as f:
    uuid_list = json.load(f)


swapped_list = []
swapped_idx = []

static_cache = 'static'
if not os.path.exists(static_cache):
    os.mkdir(static_cache)

image_cache = os.path.join(static_cache, 'image')
if not os.path.exists(image_cache):
    os.mkdir(image_cache)

@app.route("/swapped", methods=["GET"])
def has_swapped():
    # send image back
    uuid = request.args.get('uuid')
    item = request.args.get('item')
    if len(uuid_list[uuid][item]) > 0:
        return jsonify({
        "status": "Accepted",
        })
    else:
        return jsonify({
        "status": "WA",
        })

@app.route("/result", methods=["GET"])
def swapped_images():
    # send image back
    uuid = request.args.get('uuid')
    item = request.args.get('item')
    img_idx = int(request.args.get('idx'))
    print(img_idx)
    if img_idx < len(uuid_list[uuid][item]):
        #img_url = os.path.join(request.url_root, )
        uuid_cache = os.path.join(image_cache, uuid)
        re_url = url_for(uuid_cache, filename=uuid_list[uuid][item][img_idx])
        #re_url = url_for('static', filename='0.jpg')
        print(request.url_root)
        img_url = os.path.join(request.url_root, re_url)
        print(img_url)
        img_url = request.url_root + re_url[1:]
        print(img_url)
      #  path = '/home/xiaochen/Workspace/TransfiguringPortraits/backend/static/0.jpg'
      #  resp = Response(open(path, 'rb'), mimetype="image/jpeg")
      #  return resp
        return jsonify({
        "status": "Success",
        "url": img_url
        })

        #return send_file(swapped_list[img_idx], mimetype='image/jpeg')
    else:
        return jsonify({
        "status": "WI",
    })

#@app.route("/item", methods=["POST"])
#def item():
#    item = request.json["item"]
# #   swapper.request_images(item)
#    print('received item')
#
#    results = list()
# #   #for idx in range(swapper.img_engine.length()):
# #   for idx in range(1):
# #       swapped_img = swapper.process_one(idx)
# #       print(type(swapped_img))
# #       if swapped_img:
# #           results.append(swapped_img)
# #           cv2.imshow(swapped_img)
# #           cv2.waitKey(0)
#
#    return jsonify({
#        "status": "OK",
#        "result": results
#    })
#
#@app.route("/image", methods=["POST"])
#def image():
#    img = request.files["image"].read()
#    npimg = np.fromstring(img, np.int8)
#    cvimg = cv2.imdecode(npimg, 1)
#    #swapper.set_image(cvimg)
#    print('received image')
#    return jsonify({
#        "status": "OK",
#    })

@app.route("/", methods=["POST"])
def swap():
#    print(request.json)
    img = request.files["image"].read()
    item = request.form['item']
    uuid = request.form['uuid']
#    print(img)
#    print(item)

    def swapping(uuid, img, item):
        for i in range(5):
            time.sleep(10)
            
            uuid_cache = os.path.join(image_cache, uuid) 
            if not os.path.exists(uuid_cache):
                os.mkdir(uuid_cache)
                uuid_data = dict()
                uuid_data[item]= [] 
#                uuid_data['img_idx']= 0 
                uuid_list[uuid] = uuid_data

            img_name = '%s-%d.jpg' % (item, i+len(uuid_list[uuid][item]))
            img_file = os.path.join(uuid_cache, img_name)
            print(img_file)
            npimg = np.fromstring(img, np.int8)
            cvimg = cv2.imdecode(npimg, 1)
            cv2.imwrite(img_file, cvimg)
            uuid_list[uuid][item].append(img_name)
#            swapped_list.append('%d.jpg' % i)
#            swapped_list.append(img_file)

            # I need better way to save
            with open(user_cache, "w") as f:
                json.dump(uuid_list, f)
        return

        swapped_list.clear()
        npimg = np.fromstring(img, np.int8)
        cvimg = cv2.imdecode(npimg, 1)
        swapper.set_image(cvimg)
        swapper.request_images(item)

        for idx in range(swapper.img_engine.length()):
            swapped_img = swapper.process_one(idx)
            if not swapped_img:
                swapped_list.append(swapped_img)
    
    thread = threading.Thread(target=swapping, kwargs={'img': img, 'item': item})
    thread.start()

    return jsonify({
        "status": "OK",
    })

if __name__ == "__main__":
     #app.run(debug=True, host= '0.0.0.0')
     app.run(host='192.168.31.126')
     #app.run(host='127.0.0.1',port=4949, debug=True)
