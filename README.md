# AML_QuickDraw
A simple LSTM based hand drawn sketch prediction models.

## Pre-requisites
The following packages are required:
	1. Tensorflow
	2. Keras
	3. Flask
	4. Python3 
	5. Numpy

## Usage
To test the prediction models run on the shell:
```bash
python3 bidirectional.py
```
Then open your browser and enter 127.0.0.1:5000/autoencoder

To run the sequence to sequence models run s2s.py and follow the above step.

## Note 
The models were trained on google colab and then import to local machine to predict using the HTML UI. We recommend to run the training code on your colab and then test your model. 

## Acknowledgement
This project was done as a part of Advanced Machine Learning Course at PES University.