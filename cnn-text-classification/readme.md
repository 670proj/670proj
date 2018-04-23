
This is the source code for cs670 project for online music genre prediction. The basic idea of our approach is to preprocess the lyrics for given songs to feed the preprocessed word vectors the convolutional neural network, and evaluate the predictions to get a model that is able to give accurate predictions at last. Upon the completion of our model with good accuracy, the pre-trained model will be integrated in our web application for online genre prediction.


App folder contains all the source files for the web application. We have also used Python Flask to develop our backend for data storage and retrieval, and communicate with the fron-end, and the trained CNN model is included in the webpage for online classificaiton.
