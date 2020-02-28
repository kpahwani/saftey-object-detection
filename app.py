import operator
import os
from predict_function import predict
from flask import Flask, request, render_template,send_from_directory
from PIL import Image
from keras.models import load_model


app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    helmet_score= 0
    safety_gloves_score= 0
    safety_goggles=0
    safety_vest =0
    target = os.path.join(APP_ROOT, 'files')
    total_images = len(request.files.getlist("file"))
    if not os.path.isdir(target):
        os.mkdir(target)
    for upload in request.files.getlist("file"):
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
        model = load_model('models/object_detection2.model', compile=False)
        predicted_img = predict(model=model, img=image)
        helmet_score += predicted_img[0]
        safety_gloves_score += predicted_img[1]
        safety_goggles += predicted_img[2]
        safety_vest += predicted_img[3]

    predicted_images_dict = {'Helmet': (helmet_score/total_images)*100,
                            'Safety Gloves': (safety_gloves_score/total_images)*100,
                            'Safety Goggles': (safety_goggles/total_images)*100,
                            'Safety Vest': (safety_vest/total_images)*100}

    sorted_dict = dict(sorted(predicted_images_dict.items(), key=operator.itemgetter(1), reverse=True))
    print("Sorted dict", sorted_dict)
    return render_template("complete.html", parent_dict=sorted_dict)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("files", filename)


@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./images')
    print(image_names)
    return render_template("gallery.html", image_names=image_names)


if __name__ == '__main__':
    app.run(debug=True, threaded=False)