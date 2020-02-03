from tensorflow.python.keras.models import load_model
import pandas as pandas
import numpy as np

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
test_input = np.concatenate((data[40:50, 0:4], data[90:100, 0:4], data[140:150, 0:4]), axis=0)
test_output = np.concatenate((output[40:50], output[90:100], output[140:150]), axis=0)
model = load_model("model.hdf5")
predict = model.predict(x=test_input)
round = np.round(predict)
j = 0
for i in range(30):
    print(round[i])
    print(test_output[i])
    if (np.array_equal(round[i], test_output[i])):
        j += 1
print("accuracy = %", j * 100 / 30)
