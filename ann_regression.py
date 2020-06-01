import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


####################
#Data Preprocessing
####################


#Import
ds = pd.read_excel('Folds5x2_pp.xlsx')
x = ds.iloc[:, :-1].values
y = ds.iloc[:, -1].values

#Split data test and train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)

#Feature Scaling, absolutely neccesary in deep learning models***
sc = StandardScaler() 
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test) 


######################
#Build Neural Network
######################

#initializing ann
ann = tf.keras.models.Sequential()

#adding input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) #6 is random number, there is no designated number of neurons, must experimentate
#relu = rectifier activator function, sigmoid = sigmoid

#add second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # ''

#add output layer
ann.add(tf.keras.layers.Dense(units=1))
#sigmoid = predicting binary, softmax = predicting category


######################
#Train Neural Network
######################


#compile
ann.compile(optimizer='adam', loss='mean_squared_error')

#train
ann.fit(x_train, y_train, batch_size=32, epochs=100) #batch_size = number of predictions to compare with result value, epoch = # of cycles in neural networks

#predict result of a single observation (single input (independent variables from one row))
#must apply same scaling method with sc.transform()
#print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) >0.5) #1 0 0 = geography enconded variable/ France
#if result > 0.5, predicted result is 1 (meaning Yes, employee would leave company)


######################
#Predict Test Results
######################


y_pred = ann.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

