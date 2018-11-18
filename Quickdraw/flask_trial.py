from flask import Flask,render_template
from flask import request
from keras.models import load_model
import numpy as np
import json

app = Flask(__name__)

@app.route('/autoencoder')
def getCanvas():
    return render_template("canvas.html")

@app.route('/predictDiagram',methods=['POST'])
def decode_sequence():
    # Encode the input as state vectors.
    #model = load_model('autoencoder.h5')

    input_seq = np.array((request.json)['strokes'])
    max_decoder_seq_length = 129

    encoder_model = load_model('encoder.h5')
    decoder_model = load_model('decoder.h5')
    states_value = encoder_model.predict(input_seq[:max_decoder_seq_length])

    num_decoder_tokens = 5

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

        output_sequence = list(map(abs,output_sequence))
        decoded_sequence.append(output_sequence)

        # Exit condition: either hit max length
        # or find stop character.
        if ( isLastToken(output_sequence) or
           len(decoded_sequence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = output_sequence

        # Update states
        states_value = [h, c]
    print(decoded_sequence)
    print(decoded_sequence.shape)

    data = {'strokes':decoded_sequence[0]}
    return json.dumps(data) #decoded_sequence

if __name__ == "__main__":
        app.run(debug=True)
