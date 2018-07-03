This is implementation for this paper "Multi-channel LSTM-CNN model for Vietnamese sentiment analysis".

We provide a sentiment analysis dataset: VS. The dataset include two version: tokenized and without tokenized.

To run this code: 
1. Please specify a data path in preprocessing.py and load_data.py.
2. Run "python preprocessing.py" and then "python cnn_lstm.py"


Requirement:
1. Keras
2. Tensorflow

If you found this work useful, please cite it:

@INPROCEEDINGS{Quan, 
	author={Q. H. Vo and H. T. Nguyen and B. Le and M. L. Nguyen}, 
	booktitle={2017 9th International Conference on Knowledge and Systems Engineering (KSE)}, 
	title={Multi-channel LSTM-CNN model for Vietnamese sentiment analysis}, 
	year={2017},  
	pages={24-29}, 
	keywords={Analytical models;Feature extraction;Logic gates;Machine learning;Neural networks;Sentiment analysis}, 
	doi={10.1109/KSE.2017.8119429}, 
	month={Oct},
}