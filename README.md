# Predict-Your-Body-Weight-Category-with-DL

Application Link:

Demo Video Link:

## Overview:

Predict-Your-Body-Weight-Category-with-DL is an application that uses
Artificial Neural Networks (ANNs) to predict the weight category of an 
individual based on their age, height, weight, lifestyle, and habits.

The user is asked to answer 12 questions, including personal information
like age, gender, height, and weight, as well as lifestyle habits such
as family history of overweight, eating high caloric food frequently, 
smoking, monitoring calorie intake, and transportation used.

Based on these inputs, the web application uses an Artificial Neural 
Network (ANN) to predict the weight category of the user. It is a useful 
tool for anyone who wants to monitor their weight category and make 
changes to their lifestyle habits accordingly.

## Approach

The code starts by reading in the weight.csv file from the given 
file path and dropping some irrelevant columns from the data. It then 
creates dummy variables for the categorical columns and splits the
data into independent and dependent variables.

The target variable is encoded using LabelEncoder and the data is split 
into training and testing sets using train_test_split. The input 
features are then scaled using StandardScaler to normalize their values.

Next, the ANN model is defined using Sequential API from Keras. It consists 
of three layers with 32, 16, and 7 neurons, respectively. The input layer uses 
the 'relu' activation function while the output layer uses 'softmax' activation
function since the problem requires multiclass classification.

The model is then compiled using 'adam' optimizer and 'sparse_categorical_crossentropy'
loss function. Finally, the model is trained on the training set with 50 epochs 
and batch size of 32, and validated on the testing set. The accuracy of the model 
is printed out for evaluation purposes.
