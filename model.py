import numpy
import numpy as np
import os

from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

emotionsMap = {"happy": 0, "sad": 1, "chill": 2, "angry": 3, "invalid": 4}

path = "TrainingData/HighPolling/"

train_acceleration = []
train_emotion = []

input_data_array = None
label_data_array = None

for directory, subdirectories, files in os.walk(path):
    for file in files:
        if "Parsed" in file and directory != "Test":
            for emotion in emotionsMap.keys():
                if emotion in file:
                    input_data = numpy.genfromtxt(os.path.join(directory, file), dtype=float, delimiter=',', names=True)
                    target_array_shape = (45, 3)
                    pad_x = (target_array_shape[0] - input_data.shape[0])

                    input_data = [list(item) for item in input_data]
                    input_data = np.delete(input_data, 3, 1)
                    input_data = np.pad(input_data, ((pad_x, 0), (0, 0)), mode="constant")

                    input_data = input_data.flatten()

                    if input_data_array is None:
                        input_data_array = input_data
                    else:
                        input_data_array = np.vstack((input_data_array, input_data))

                    if label_data_array is None:
                        label_data_array = emotionsMap[emotion]
                    else:
                        label_data_array = np.vstack((label_data_array, emotionsMap[emotion]))

                    break

print(input_data_array.shape)
input_data_array = np.array(input_data_array, dtype="float32")
print(type(input_data_array[0][0]))
one_hot_vectors = torch.nn.functional.one_hot(torch.tensor(label_data_array).squeeze(), num_classes=5)
label_data_array = np.array(one_hot_vectors, dtype="float32")
print(type(label_data_array[0].shape))
print(type(label_data_array[0][0]))


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()

        # Define the hidden layer
        self.hidden = nn.Linear(input_size, hidden_size)

        # Define the output layer
        self.output = nn.Linear(hidden_size, output_size)

        # Define the activation function for the hidden layer
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through the hidden layer with ReLU activation
        x = self.relu(self.hidden(x))

        # Forward pass through the output layer
        x = self.output(x)

        return x


input_size = 135
hidden_size = 8
output_size = 5

model = SimpleNN(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

input_data = torch.Tensor(input_data_array)
labels = torch.Tensor(label_data_array)

num_epochs = 7000

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(input_data)
    _, predicted = torch.max(outputs, 1)

    loss = criterion(outputs, labels)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

test_path = "TrainingData/HighPolling/Test/"

test_data_array = None
test_labels = None

for directory, subdirectories, files in os.walk(test_path):
    for file in files:
        if "Parsed" in file:
            input_data = numpy.genfromtxt(os.path.join(directory, file), dtype=float, delimiter=',', names=True)
            target_array_shape = (45, 3)
            pad_x = (target_array_shape[0] - input_data.shape[0])

            input_data = [list(item) for item in input_data]
            input_data = np.delete(input_data, 3, 1)
            input_data = np.pad(input_data, ((pad_x, 0), (0, 0)), mode="constant")

            input_data = input_data.flatten()

            if test_data_array is None:
                test_data_array = input_data
            else:
                test_data_array = np.vstack((test_data_array, input_data))

            if test_labels is None:
                test_labels = emotionsMap[emotion]
            else:
                test_labels = np.vstack((test_labels, emotionsMap[emotion]))

torch.save(model, 'trained_model.pth')

print(test_data_array.shape)
test_data_array = np.array(test_data_array, dtype="float32")
print(type(test_data_array[0][0]))
test_data_array = torch.Tensor(test_data_array)
test_labels = torch.Tensor(test_labels)

with torch.no_grad():
    model.eval()
    test_outputs = model(test_data_array)
    _, predicted = torch.max(test_outputs, 1)
