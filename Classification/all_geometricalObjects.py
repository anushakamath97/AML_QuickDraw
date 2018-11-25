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

import keras
from keras.layers import Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
import numpy as np
from sklearn.model_selection import train_test_split

'''
#basic LSTM
model = Sequential()
model.add(LSTM(50,input_shape=(100,2)))
model.add(Dense(5, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.load_weights("model/Geometrical_Objects/model_Lstm.h5")
'''


#1 conv layer
inputs = Input(shape=(100,2))
conv1 = Conv1D(5,2,activation='relu',padding='same')(inputs)
lstm = LSTM(50)(conv1)
dense = Dense(5, activation='softmax')(lstm)
model = Model(inputs,dense)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.load_weights("model/Geometrical_Objects/model_convLstm1.h5")


'''
#2 conv layer
inputs = Input(shape=(100,2))
conv1 = Conv1D(5,2,activation='relu',padding='same')(inputs)
conv2 = Conv1D(5,2,activation='relu',padding='same')(conv1)
lstm = LSTM(50)(conv2)
dense = Dense(5, activation='softmax')(lstm)
model = Model(inputs,dense)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.load_weights("model/Geometrical_Objects/model_convLstm2.h5")
'''

'''
#cnn
model = Sequential()
model.add(Conv1D(32, kernel_size=2,
                 activation='relu',
                 input_shape=(100,2)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print(model.summary())
model.load_weights("model/Geometrical_Objects/model_cnn.h5")
'''


app = Flask(__name__)

@app.route('/predict', methods = ['POST','GET'])
def prediction():
    data = request.json
    data = data["stroke"]
    if(len(data)<100):
        for j in range(0,100-len(data)):
            data.append((0,0))
    data_to_predict = [data]
    data_to_predict = np.array(data_to_predict)
    print(data_to_predict.shape)
    pred = model.predict(data_to_predict)
    print(pred)
    label = np.argmax(pred[0])
    if(label==0):
        value = "square"
    elif(label==1):
        value = "line"
    elif(label==2):
        value = "hexagon"
    elif(label==3):
        value = "circle"
    else:
        value= "triangle"
    result = {"result":value}
    return json.dumps(result)

@app.route("/")
def dashboard():
    return render_template("canvas_allGeometricalObjects.html")


if __name__ == "__main__":
    app.run()
