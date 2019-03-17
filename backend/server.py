from flask import Flask
from image_engine import BingImageEngine


app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello world!"

if __name__ == "__main__":
    app.run()