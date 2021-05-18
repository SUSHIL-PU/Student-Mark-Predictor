import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load("model/student_mark_predictor.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        input_features = [float(x) for x in request.form.values()]
        features_value  = np.array(input_features)
    except:
        return render_template('index.html', prediction_text='Something Went Wrong!')
    
    #validate input hours
    if input_features[0] <= 0 or input_features[0] > 24:
        return render_template('index.html', prediction_text='Enter valid hours per day')

    # Predicting the output from the featured_value
    output = model.predict([features_value])[0][0].round(2)

    return render_template('index.html', prediction_text='You will get [{}%] marks, if you study [{}] hours per day '.format(output, float(features_value[0])))


if __name__ == "__main__":
    app.run()    # To run the application on local development server
     