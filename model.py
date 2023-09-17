import numpy
import numpy as np
import os

import torch
# import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset, DataLoader
# import dask.dataframe as dd
import pandas as pd
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
from sklearn import preprocessing

HEIGHT = 45
WIDTH = 4
MULTIPLICATION = HEIGHT * WIDTH

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(MULTIPLICATION, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, MULTIPLICATION))


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.hidden = nn.ModuleList()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)

# happyList = []
# sadList = []
# angryList = []
# chillList = []
# invalidList = []

emotionsMap = {"happy": 0, "sad": 1, "angry": 2, "chill": 3, "invalid": 4}
# emotions = ["happy", "sad", "angry", "chill", "invalid"]

path = "TrainingData/HighPolling/"

train_acceleration = []
train_emotion = []

# dataset = []
#
# for directory, subdirectories, files in os.walk(path):
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


# print(input_data_array)
print(input_data_array.shape)
input_data_array = np.array(input_data_array, dtype="float32")
print(type(input_data_array[0][0]))
# input_data_tensor = torch.LongTensor
one_hot_vectors = torch.nn.functional.one_hot(torch.tensor(label_data_array).squeeze(), num_classes=5)
label_data_array = np.array(one_hot_vectors, dtype="float32")
print(type(label_data_array[0].shape))
print(type(label_data_array[0][0]))


import torch
import torch.nn as nn
import torch.optim as optim


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
hidden_size = 4
output_size = 5

model = SimpleNN(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming your input data is stored in a variable named 'input_data'
# and your labels are stored in a variable named 'labels'

input_data = torch.Tensor(input_data_array)
labels = torch.Tensor(label_data_array)  # Convert labels to long type

# Set the number of training epochs
num_epochs = 5000

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(input_data)

    # Compute the loss
    # print(type)
    loss = criterion(outputs, labels)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss at each epoch (optional)
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

with torch.no_grad():
    model.eval()
    test_output = model(test_data)