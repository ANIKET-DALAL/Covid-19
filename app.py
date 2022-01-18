import os
from flask import Flask, render_template, redirect, request
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

app = Flask(__name__)

model = load_model("model_adv.h5")

@app.route('/')
def hello():
	return render_template("index.html")


@app.route('/submit', methods = ["POST"])
def submit_data():
	if request.method == "POST":

		f = request.files['userfile']
		path = "./static/{}".format(f.filename)
		f.save(path)

		test_image = image.load_img(path,target_size=(224,224))
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis = 0)
		result = (model.predict(test_image) > 0.5).astype("int32")
		if result[0][0] == 0:
        
			prediction = 'Patient is Covid Positive'
		else:
			prediction = 'Patient is Covid Negative'

	return render_template("result.html",image_name=f.filename, text=prediction)

@app.route('/covid-19')
def about_covid():
	return render_template("covid-19.html")

@app.route('/model')
def about_model():
	return render_template("model.html")

if __name__ == '__main__':
	app.run(debug = True)