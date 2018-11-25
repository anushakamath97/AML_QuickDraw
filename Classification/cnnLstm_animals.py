# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 09:13:19 2018

@author: Aishwarya
"""

#tensorflowjs_converter --input_format keras C:\xampp\htdocs\AmlProject\model\model_conv.h5 C:\xampp\htdocs\AmlProject\model\

from flask import Flask, render_template,request,send_file
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Conv1D
from keras.models import Model
import json

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split


avg_pt = 208
no_classes = 7
inputs = Input(shape=(avg_pt,2))
conv1 = Conv1D(8,2,activation='relu',padding='same')(inputs)
conv2 = Conv1D(16,2,activation='relu',padding='same')(conv1)
lstm1 = LSTM(50)(conv2)
lstm2 = RepeatVector(avg_pt)(lstm1)
lstm3 = LSTM(50)(lstm2)
dense = Dense(no_classes, activation='softmax')(lstm3)
model = Model(inputs,dense)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.load_weights("model/Animals/model_convLstm.h5")

app = Flask(__name__)

@app.route('/predict', methods = ['POST','GET'])
def prediction():
    data = request.json
    data = data["stroke"]
    if(len(data)>208):
        data = data[:208]
    if(len(data)<208):
        for j in range(0,208-len(data)):
            data.append((0,0))
    data_to_predict = [data]
    data_to_predict = np.array(data_to_predict)
    print(data_to_predict.shape)
    pred = model.predict(data_to_predict)
    print(pred)
    label = np.argmax(pred[0])
    if(label==0):
        value = "ant"
    elif(label==1):
        value = "bear"
    elif(label==2):
        value = "bird"
    elif(label==3):
        value = "cat"
    elif(label==4):
        value = "dog"
    elif(label==5):
        value = "butterfly"
    else:
        value= "fish"
    result = {"result":value}
    return json.dumps(result)

@app.route("/")
def dashboard():
    return render_template("canvas_cnn_animals.html")


if __name__ == "__main__":
    app.run()
