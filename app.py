#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 17:06:40 2021

@author: Dippies
"""
import os
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import joblib
import pickle

#os.chdir('/users/Dippies/Downloads')
#print(os.getcwd())

app = Flask(__name__, template_folder='template')
#model = pickle.load(open('model.pkl', 'rb'))
model = joblib.load('linreg.pkl')
# takes 3 params

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        print(message)
        print(data)
#        vect = cv.transform(data).toarray()
        my_prediction = model.predict(data)
        return render_template('prediction_outcome.html', prediction = my_prediction)


# =============================================================================

if __name__ == "__main__":
    app.run(debug=False)
    
    