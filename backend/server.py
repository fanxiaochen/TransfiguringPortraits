from flask import Flask, request, jsonify
from image_engine import BingImageEngine
from swapper import *


app = Flask(__name__)

swapper = Swapper()
swapper.start_img_engine()
#swapper.start_fs_engine()

@app.route("/item", methods=["POST"])
def item():
    print(request.json)
    item = request.json["item"]

@app.route("/image", methods=["POST"])
def image():
    img = request.files.get('image')
    return jsonify({
        "status": "OK",
    })

@app.route("/", methods=["POST"])
def swap():
    print(request.json)
    image = request.json["image"]
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