from flask import Flask, request, jsonify
from image_engine import BingImageEngine
from swapper import *


app = Flask(__name__)

swapper = Swapper()
swapper.start_img_engine()
#swapper.start_fs_engine()


@app.route("/", methods=["POST"])
def swap():
    image = request.json["image"]
    item = request.json["item"]
    swapper.set_image(image)
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