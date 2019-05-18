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

swapper = Swapper()
swapper.start_img_engine()
swapper.start_fs_engine()

user_cache = 'user.json'
if not os.path.exists(user_cache):
    with open(user_cache,'w') as f:
        json.dump(dict(),f)

with open(user_cache,"r") as f:
    uuid_list = json.load(f)


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
    if uuid in uuid_list and item in uuid_list[uuid] and len(uuid_list[uuid][item]) > 0:
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
        img_url = os.path.join(image_cache, uuid, uuid_list[uuid][item][img_idx])
        print(img_url)
        
        #uuid_cache = os.path.join('image', uuid)
        #re_url = url_for('static', filename=os.path.join(uuid_cache, uuid_list[uuid][item][img_idx]))
        #re_url = url_for('static', filename='0.jpg')
        print(request.url_root)
        img_url = os.path.join(request.url_root, img_url)
      #  print(img_url)
      #  img_url = request.url_root + re_url[1:]
      #  print(img_url)
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

@app.route("/", methods=["POST"])
def swap():
#    print(request.json)
    img = request.files["image"].read()
    item = request.form['item']
    uuid = request.form['uuid']
    print(uuid)
    print(item)

    def swapping(uuid, img, item):
#        for i in range(5):
#            time.sleep(10)
#            
#            uuid_cache = os.path.join(image_cache, uuid) 
#            if not os.path.exists(uuid_cache):
#                os.mkdir(uuid_cache)
#                uuid_data = dict()
#                uuid_data[item]= [] 
##                uuid_data['img_idx']= 0 
#                uuid_list[uuid] = uuid_data
#
#            img_name = '%s-%d.jpg' % (item, len(uuid_list[uuid][item]))
#            img_file = os.path.join(uuid_cache, img_name)
#            print(img_file)
#            npimg = np.fromstring(img, np.int8)
#            cvimg = cv2.imdecode(npimg, 1)
#            cv2.imwrite(img_file, cvimg)
#            uuid_list[uuid][item].append(img_name)
##            swapped_list.append('%d.jpg' % i)
##            swapped_list.append(img_file)
#
#            # I need better way to save
#            with open(user_cache, "w") as f:
#                json.dump(uuid_list, f)
#        return

        uuid_cache = os.path.join(image_cache, uuid) 
        if not os.path.exists(uuid_cache):
            os.mkdir(uuid_cache)
            uuid_data = dict()
            uuid_data[item]= [] 
#                uuid_data['img_idx']= 0 
            uuid_list[uuid] = uuid_data
            # I need better way to save
            with open(user_cache, "w") as f:
                json.dump(uuid_list, f)

        npimg = np.fromstring(img, np.int8)
        cvimg = cv2.imdecode(npimg, 1)
        img_h, img_w, _ = cvimg.shape
        scale = int(img_w / 1000) + 4
        resized = (int(img_w / scale), int(img_h /scale))
        print(resized)
        scaled_img = cv2.resize(cvimg, resized)
        time_start = time.time()
        src_img = swapper.set_image(scaled_img)
        time_end = time.time()
        print('set source image time:', time_end-time_start)

        time_start = time.time()
        tgt_imgs = swapper.request_images(item)
        time_end = time.time()
        print('request images time:', time_end-time_start)

        for idx in range(len(tgt_imgs)):
            time_start = time.time()
            tgt_img = swapper.set_image(tgt_imgs[idx])
            time_end = time.time()
            print('set target image time:', time_end-time_start)

            time_start = time.time()
            swapped_img = swapper.process_one(src_img, tgt_img)
            time_end = time.time()
            print('process_one time:', time_end-time_start)
            if swapped_img.size != 0:
                img_name = '%s-%d.jpg' % (item, len(uuid_list[uuid][item]))
                img_file = os.path.join(uuid_cache, img_name)
                cv2.imwrite(img_file, swapped_img)
                uuid_list[uuid][item].append(img_name)
                # I need better way to save
                with open(user_cache, "w") as f:
                    json.dump(uuid_list, f)
        return
    
    thread = threading.Thread(target=swapping, kwargs={'uuid': uuid, 'img': img, 'item': item})
    thread.start()

    return jsonify({
        "status": "OK",
    })

if __name__ == "__main__":
     #app.run(debug=True, host= '0.0.0.0')
     app.run(host='192.168.31.126')
     #app.run(host='127.0.0.1',port=4949, debug=True)
