# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
#http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/



from keras.models import Sequential
from keras.layers import Dense

import numpy

seed = 7
numpy.random.seed(seed)

#load data
dataset = numpy.loadtxt("pima_indians_diabetes.data", delimiter=",")

#split into input X and output Y variable
X = dataset[:, 0:8]
Y = dataset[:, 8]

#create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

#compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the model
model.fit(X, Y, epochs=150, batch_size=10)

#Evaluate the model
scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#calculate predictions
predictions = model.predict(X)

#round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

