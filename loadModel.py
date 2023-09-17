import os

import torch
import model
import numpy as np
from torch import nn
#
# # define any number of nn.Modules (or use your current ones)
# encoder = nn.Sequential(nn.Linear(45 * 5, 64), nn.ReLU(), nn.Linear(64, 3))
# decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 5 * 45))

# load checkpoint
checkpoint = "./lightning_logs/version_18/checkpoints/epoch=19-step=2000.ckpt"
autoencoder = model.LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=model.encoder, decoder=model.decoder)
# autoencoder = model.LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)
autoencoder = autoencoder.to('cpu')

# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

# test_acceleration = []

test_path = "TrainingData/HighPolling/Test"
for directory, subdirectories, files in os.walk(model.path):
    for file in files:
        if "Parsed" in file:
            data = np.genfromtxt(os.path.join(directory, file), dtype=float, delimiter=',', names=True)
            target_array_shape = (45, 4)
            pad_x = (target_array_shape[0]-data.shape[0])

            flat = []
            data = [list(item) for item in data]

            # print(data)
            # if len(data[0]) == 5:
                # print(data)
            data = np.pad(data, ((pad_x, 0), (0, 0)), mode="constant")

            for element in data:
                for nested in element:
                    flat.append(nested)

            test_acceleration = np.array(flat)
            tensor_acceleration = torch.Tensor(test_acceleration).to('cpu')

            embeddings = encoder(tensor_acceleration)
            print(f'{os.path.join(directory, file)}: {embeddings}')
            # prob = nn.functional.softmax(output_en, dim=1)


            ###################

            #flatten embeddings
            embeddings_flat = embeddings.flatten()

            # Get the index of the maximum value
            test, predicted_index = torch.max(embeddings_flat, 0)

            # Convert the index to an integer
            digit = int(predicted_index.item())

            #return the result
            print(digit)