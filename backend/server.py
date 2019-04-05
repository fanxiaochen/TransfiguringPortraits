from flask import Flask, request, jsonify
import numpy as np
import cv2
from image_engine import BingImageEngine
from swapper import *


app = Flask(__name__)

swapper = Swapper()
swapper.start_img_engine()
swapper.start_fs_engine()

@app.route("/item", methods=["POST"])
def item():
    item = request.json["item"]
    swapper.request_images(item)

    results = list()
    #for idx in range(swapper.img_engine.length()):
    for idx in range(5):
        swapped_img = swapper.process_one(idx)
        print(type(swapped_img))
        if swapped_img:
            results.append(swapped_img)
            cv2.imshow(swapped_img)
            cv2.waitKey(0)

    return jsonify({
        "status": "OK",
        "result": results
    })

@app.route("/image", methods=["POST"])
def image():
    img = request.files["image"].read()
    npimg = np.fromstring(img, np.int8)
    cvimg = cv2.imdecode(npimg, 1)
    swapper.set_image(cvimg)
    return jsonify({
        "status": "OK",
    })

@app.route("/", methods=["POST"])
def swap():
    print(request.json)
    img = request.files["image"].read()

    npimg = np.fromstring(img, np.int8)
    cvimg = cv2.imdecode(npimg, 1)
    item = request.json["item"]
    #swapper.set_image(image)
    swapper.request_images(item)

    results = list()
    for idx in range(swapper.img_engine.length()):
        pass
     #   swapped_img = swapper.process_one(idx)
     #   if not swapped_img:
     #       results.append(swapped_img)

    return jsonify({
        "status": "OK",
        "result": results
    })

if __name__ == "__main__":
    app.run()