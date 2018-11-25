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


#basic bidirectional RNN
model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True),
                        input_shape=(400,2)))
model.add(Bidirectional(LSTM(20)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
print(model.summary())

model.load_weights("model/Animals/model_bidirectional.h5")

app = Flask(__name__)

@app.route('/predict', methods = ['POST','GET'])
def prediction():
    data = request.json
    data = data["stroke"]
    if(len(data)<400):
        for j in range(0,400-len(data)):
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
        value = "cat"
    elif(label==3):
        value = "butterfly"
    else:
        value= "fish"
    result = {"result":value}
    return json.dumps(result)

@app.route("/")
def dashboard():
    return render_template("canvas_BidirectionalLstm_animal.html")


if __name__ == "__main__":
    app.run()
