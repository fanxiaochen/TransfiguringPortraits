import threading
from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
from image_engine import BingImageEngine
from swapper import *


app = Flask(__name__)

swapper = Swapper()
swapper.start_img_engine()
swapper.start_fs_engine()

swapped_list = []
swapped_idx = 0

@app.route("/swapped", methods=["GET"])
def has_swapped():
    # send image back
    if len(swapped_list) > 0:
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
    img_idx = request.args.get('img_idx')
    if img_idx < len(swapped_list):
        return send_file(swapped_list[img_idx], mimetype='image/jpeg')
    else:
        return jsonify({
        "status": "WI",
    })

@app.route("/item", methods=["POST"])
def item():
    item = request.json["item"]
 #   swapper.request_images(item)
    print('received item')

    results = list()
 #   #for idx in range(swapper.img_engine.length()):
 #   for idx in range(1):
 #       swapped_img = swapper.process_one(idx)
 #       print(type(swapped_img))
 #       if swapped_img:
 #           results.append(swapped_img)
 #           cv2.imshow(swapped_img)
 #           cv2.waitKey(0)

    return jsonify({
        "status": "OK",
        "result": results
    })

@app.route("/image", methods=["POST"])
def image():
    img = request.files["image"].read()
    npimg = np.fromstring(img, np.int8)
    cvimg = cv2.imdecode(npimg, 1)
    #swapper.set_image(cvimg)
    print('received image')
    return jsonify({
        "status": "OK",
    })

@app.route("/", methods=["POST"])
def swap():
#    print(request.json)
    img = request.files["image"].read()
    item = request.form['item']
#    print(img)
#    print(item)

    def swapping(img, item):
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
