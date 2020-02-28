
import os

from PIL import Image
from flask import Flask, request, render_template, send_from_directory, redirect
from keras.models import load_model

from predict_function import predict

app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    predicted_images_dict={}
    target = os.path.join(APP_ROOT, 'files')
    if not os.path.isdir(target):
        os.mkdir(target)

    for key, upload in request.files.to_dict().items():
        filename = upload.filename
        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if (ext == ".jpg") or (ext == ".png"):
            print("File supported moving on...")
        else:
            render_template("Error.html", message="Files uploaded are not supported...")
        destination = "/".join([target, filename])
        upload.save(destination)
        image = Image.open(destination)
        image = image.resize(size=(299, 299))
        model = load_model('models/object_detection_1.model', compile=False)
        predicted_img = predict(model=model, img=image)
        predicted_images_dict[filename] = {'helmet': predicted_img[0], 'gloves': predicted_img[1]}

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete.html", parent_dict=predicted_images_dict)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("files", filename)


@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./images')
    return render_template("gallery.html", image_names=image_names)


if __name__ == '__main__':
    app.run(debug=True, threaded=False)
