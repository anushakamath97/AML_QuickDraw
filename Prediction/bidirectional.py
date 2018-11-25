from flask import Flask,render_template
from flask import request
from keras.models import load_model
import tensorflow as tf
import random
import numpy as np
import json

app = Flask(__name__)

seq_lengths = { 'cat' : 129, 'square': 34, 'airplane' : 99} 

def isLastToken(token):
	for i in range(4):
		if token[i] != 0:
			return False
	if token[4] == 1:
		return True
	return False

@app.route('/autoencoder')
def getCanvas():  
    return render_template("canvas.html")

@app.route('/getRandom')
def getFromData():
	model = request.args["model"]
	data = np.load("data/"+model+".npy")
	num = random.randint(0,data.shape[0]-1)
	output_seq = list(map(lambda point: list(map(lambda x: int(round(x)),point)), data[num]))
	sending = {'strokes':output_seq, 'prev' : [0,0]}
	return json.dumps(sending)

@app.route('/predictDiagram',methods=['POST'])
def decode_sequence():
	req = request.json
	input_seq = np.array(req['strokes'])
	prev = req['prev']
	
	max_decoder_seq_length = seq_lengths[req['model']]
	with tf.Session() as sess:
		model = load_model("models/bidirectional_"+req['model']+".h5")
		model.compile(optimizer='adam',loss = 'mean_squared_error', metrics=['accuracy'])
		stop_condition = False
		output = model.predict(input_seq)
		index = output.shape[1]
		stop_condition = False
		output_sequence = output[0]
		output_sequence = list(map(lambda point: list(map(lambda x: int(round(x)),point)), output_sequence))
		while(stop_condition != True) :
			token = model.predict(np.array([[output_sequence[index-1]]]))
			if (isLastToken(token[0][0]) or len(output_sequence) > max_decoder_seq_length):
				stop_condition = True
				break
			token = list(map(lambda x: int(round(x)),token[0][0]))
			output_sequence.append(token)
			index += 1
	data = {'strokes':output_sequence, 'prev':prev}
	return json.dumps(data) #decoded_sequence

if __name__ == "__main__":
        app.run(debug=True)
