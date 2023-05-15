# save this as app.py
from flask import Flask, escape, request, render_template , send_from_directory
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# model = load_model("models/fruits.h5")
model = load_model("models/1")

class_name = ['freshapples', 'freshbanana', 'freshoranges',
              'rottenapples', 'rottenbanana', 'rottenoranges']

app = Flask(__name__)

#routing
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        f = request.files['fruit']
        filename = f.filename
        target = os.path.join(APP_ROOT, 'images/')
        print(target)
        des = "/".join([target, filename])
        f.save(des)

        test_image = load_img(
            "images\\"+filename, target_size=(256, 256))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        prediction = model.predict(test_image)
        print(prediction)

        predicted_class = class_name[np.argmax(prediction[0])]
        print(predicted_class)
        confidence = np.max(prediction[0])
        print(confidence)

        return render_template("prediction.html",image_name = filename, confidence="chance -> "+str(confidence), prediction="prediction -> "+str(predicted_class))

    else:
        return render_template("prediction.html")
   

@app.route('/prediction/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)



if __name__ == "__main__":
    app.debug = True
    app.run()