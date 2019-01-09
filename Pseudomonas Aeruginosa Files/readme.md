# Run Instructions:
## Training (Generates a new model):
place validation images to be classified in /images/validation
run these commands from this directory:
pip install ...
	os
	fnmatch
	pandas
	random
	numpy
	PIL
	glob
	sys
	tensorflow
python preprocessing.py
python load_dataset.py
python NN_project.py

##### training explanation:
We are using tensorflow to train our neural network
First this requires python to have all the necessary modules to run our code
Then we process the data to fit a 224x224 image size. These images also have a chance to be rotated and flipped randomly, which augments the number of samples in the training set.
Images have been cleaned to remove examples with excessive gridding or visible writing
Also we record strain numbers and corresponding carb.auc.delta* values, and label them based on the 75% quantile for this feature.

We then use the load_dataset script to connect our target labels (carb.auc.delta) to our image samples, and place them in Tensorflow datasets

Images are classified by the 75% quantile of the dataset. If a carbenicillin change is less than the 75% quantile for the dataset, the image is classified as 0, or weak to the antibiotic
otherwise, the image is classified as 1, or resistant to the antibiotic.

Then we use the NN_project.script to actually construct and train our model. This utilizes the dataset contructed in load_dataset and loads in a 10-layer 
network with relu activation functions and a learning rate of 0.0001. Convolutional, pooling, and dropout layers are used.

Make sure to place your validation images into /images/validation, if no images supplied, final classification accuracy will be 0

*this is from our slightly modified excel file which has a carb.auc.delta column in the total database instead of just the 'Isolates' sheet
