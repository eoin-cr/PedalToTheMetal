import os

import torch
import matplotlib.pyplot as plt
import model
import numpy as np
from torch import nn
#
# # define any number of nn.Modules (or use your current ones)
# encoder = nn.Sequential(nn.Linear(45 * 5, 64), nn.ReLU(), nn.Linear(64, 3))
# decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 5 * 45))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load checkpoint
checkpoint = "./lightning_logs/version_38/checkpoints/epoch=49-step=5000.ckpt"
autoencoder = model.LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=model.encoder, decoder=model.decoder)
# autoencoder = model.LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)
autoencoder = autoencoder.to(device)

# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

# test_acceleration = []

test_path = "TrainingData/HighPolling/Test/"
# for directory, subdirectories, files in os.walk(test_path):
for directory, subdirectories, files in os.walk(model.path):
    for file in files:
        # if "Parsed" in file:
        if "Parsed" in file and directory != "Test":
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
            tensor_acceleration = torch.Tensor(test_acceleration).to(device)

            # embeddings = encoder(tensor_acceleration)
            with torch.no_grad():
                embeddings = autoencoder.encoder(tensor_acceleration.unsqueeze(0))

            print(f'{os.path.join(directory, file)}: {embeddings}')
            # prob = nn.functional.softmax(output_en, dim=1)

            embeddings = embeddings.view(embeddings.size(0), -1)

            predicted_probs = torch.softmax(embeddings, dim=1)
            _, predicted_label = torch.max(predicted_probs, dim=1)

            predicted_label = predicted_label.squeeze().tolist()

            print(f'Predicted: {predicted_label}')