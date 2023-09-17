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

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

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

# csvMap = {"happy": [], "sad": [], "angry": [], "chill": [], "invalid": []}
emotions = ["happy", "sad", "angry", "chill", "invalid"]

path = "TrainingData/HighPolling/"

train_acceleration = []
train_emotion = []

# dataset = []
#
# for directory, subdirectories, files in os.walk(path):
for directory, subdirectories, files in os.walk(path):
    for file in files:
        if "Parsed" in file and directory != "Test":
            for emotion in emotions:
                if emotion in file:
                    # csvMap[key].append(pd.read_csv)
                    # data_df = pd.read_csv(os.path.join(directory, file))
                    # data_tensor = torch.tensor(data_df.values, dtype=torch.float32)
                    data = numpy.genfromtxt(os.path.join(directory, file), dtype=float, delimiter=',', names=True)
                    # print(data)
                    target_array_shape = (45, 4)
                    pad_x = (target_array_shape[0]-data.shape[0])
                    # print(f'{pad_x}, {data.shape[0]}')


                    # data = np.pad(data, pad_width=((45-len(data), 0), (0, 0)))
                    data = [list(item) for item in data]
                    if len(data[0]) == 5:
                        print(data)
                    data = np.pad(data, ((pad_x, 0), (0,0)), mode="constant")

                    # print(len(data))
                    train_acceleration.append(data)
                    train_emotion.append(emotion)
                    # csvMap[emotion].append(data_tensor)
                    break

train_acceleration = np.array(train_acceleration)
# print(train_acceleration)
print(f'{len(train_acceleration)}, {len(train_emotion)}')
tensor_acceleration = torch.Tensor(train_acceleration)
le = preprocessing.LabelEncoder()
targets = le.fit_transform(emotions)
tensor_emotion = torch.Tensor(targets)

# for emotion in csvMap.keys():
#     for i in range(len(csvMap[emotion])):
#         dataset.append((csvMap[emotion][i], emotion))
#         # csvMap[key][i] = torch.Tensor(csvMap[key][i])



# # setup data
# out = MNIST(os.getcwd(), download=True, transform=ToTensor())
# print(f'{len(out)}\n{out[0]} \n ----------- \n {out[1]}')
# dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
#
# batch_size = 32
# num_workers = 4
#
# train_loader = utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

dataset = torch.utils.data.TensorDataset(tensor_acceleration, tensor_emotion)
print(f'{len(tensor_acceleration)}, {len(tensor_emotion)}')
dataloader = DataLoader(dataset)
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=dataloader)

# load checkpoint
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

# # choose your trained nn.Module
# encoder = autoencoder.encoder
# encoder.eval()

class MyModel(LightningModule):
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

# embed test walk
# test_batch = []
test_path = "TrainingData/HighPolling/Test/"
for directory, subdirectories, files in os.walk(test_path):
    for file in files:
        if "Parsed" in file:
            for emotion in csvMap.keys():
                if emotion in file:
                    # csvMap[key].append(pd.read_csv)
                    data_df = pd.read_csv(os.path.join(directory, file))
                    data_tensor = torch.tensor(data_df.values, dtype=torch.float32)
                    # csvMap[key].append(data_tensor)
                    break

data_loader = utils.data.DataLoader(test_batch, batch_size=batch_size, shuffle=True, num_workers=num_workers)
model = MyModel()
trainer = LitAutoEncoder()
predictions = trainer.predict(model, data_loader)
print(predictions)
