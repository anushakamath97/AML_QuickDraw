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
    # Encode the input as state vectors.
    req = request.json

    max_decoder_seq_length = seq_lengths[req['model']]

    with tf.Session() as sess:
    	encoder_model = load_model('models/s2s_encoder.h5')
    	input_seq = np.array(req['strokes'])
    

    	#compile the model
    	encoder_model.compile(optimizer='rmsprop',loss = 'mean_squared_error', metrics=['accuracy'])
    	states_value = encoder_model.predict(input_seq)
   
    num_decoder_tokens = 5

    with tf.Session() as sess:
	    decoder_model = load_model('models/s2s_decoder.h5')
	    decoder_model.compile(optimizer='rmsprop',loss = 'mean_squared_error', metrics=['accuracy'])

	    # Generate empty target sequence of length 1.
	    target_seq = np.zeros((1, 1, num_decoder_tokens))

	    # Populate the first character of target sequence with the start character.
	    target_seq[0, 0, 2] = 1.

	    # Sampling loop for a batch of sequences
	    # (to simplify, here we assume a batch of size 1).
	    stop_condition = False
	    decoded_sequence = []
	    while not stop_condition:
	    	output_sequence, h, c = decoder_model.predict(
		    [target_seq] + states_value)

		
	    	token = list(map(lambda x: int(round(x)),output_sequence[0][0]))
		
	    	decoded_sequence.append(token)

		# Exit condition: either hit max length
		# or find stop character.
	    	if ( isLastToken(token) or
	    		len(decoded_sequence) > max_decoder_seq_length):
	    			stop_condition = True

		# Update the target sequence (of length 1).
	    	target_seq = output_sequence

		# Update states
	    	states_value = [h, c]
   
    data = {'strokes':decoded_sequence, 'prev': req['prev']}
    return json.dumps(data) #decoded_sequence

if __name__ == "__main__":
        app.run(debug=True)
