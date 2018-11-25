# AML_QuickDraw
Here, we demonstrate the classification of simple drawing objects

## Pre-requisites
The following packages are required:<br />
	1. Tensorflow <br />
	2. Keras <br />
	3. Flask <br />
	4. Python3 <br />
	5. Numpy <br />

## Usage
To test the classification models run on the shell:
```bash
python3 cnnLstm_animals.py
```
Then open your browser and enter 127.0.0.1:5000/

To run the bidirectionalLSTm model for animals type python3 bidirectional\_lstm\_animals.py and follow the above step.

To run the models for Geometrical Objects type python3 all\_all_geometricalObjects.py and open the browser.

## Note 
The models were trained on google colab and then imported to local machine to predict using the HTML UI. We recommend to run the training code on your colab and then test your model. 

## Acknowledgement
This project was done as a part of Advanced Machine Learning Course at PES University.
