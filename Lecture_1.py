# Target : build 1 model ML contains : input, hidden layers, output
# Resolve Iris problem
# Input is 1 vector 1*4
# Output is 1 vector 1 * 3 : 100,010,001
# Use Gradient Descent to optimize Loss function
import pandas as pandas
import numpy as np
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.backend import sigmoid
from tensorflow.python.keras.layers import Dense

# build model
from tensorflow.python.keras.losses import MSE
from tensorflow.python.keras.optimizers import SGD

input = Input(shape=(4,))
layer1 = Dense(units=10, activation=relu)(input)
layer2 = Dense(units=20, activation=relu)(layer1)
output = Dense(units=3, activation=sigmoid)(layer2)
model = Model(input, output)

model.summary()  # show model struct

# setup train model
model.compile(optimizer=SGD(lr=0.02), loss=MSE, metrics=['accuracy'])
# train model
# prepare data
# read file use pandas
data = np.array(pandas.read_csv(filepath_or_buffer="iris.data", header=None, nrows=150))
output = []
train_output = []
for i in range(150):
    if (data[i, 4] == "Iris-setosa"):
        output.append([1, 0, 0])
    if (data[i, 4] == "Iris-versicolor"):
        output.append([0, 1, 0])
    if (data[i, 4] == "Iris-virginica"):
        output.append([0, 0, 1])

train_input = np.concatenate((data[0:40, 0:4], data[50:90, 0:4], data[100:140, 0:4]), axis=0)
train_output = np.concatenate((output[0:40], output[50:90], output[100:140]), axis=0)
test_input = np.concatenate((data[40:50, 0:4], data[90:100, 0:4], data[140:150, 0:4]), axis=0)
print(train_input.shape)
print(train_output.shape)

model.fit(x=train_input, y=train_output, epochs=10000)
model.save("model.hdf5")

# predict = model.predict(x=test_input)
#
# # Get the maximum values of each column i.e. along axis 0
# maxInColumns = np.amax(predict, axis=0)
# print('Max value of every column: ', maxInColumns)
# print(predict)
