import os

import torch
import model
import numpy as np


# load checkpoint
checkpoint = "./lightning_logs/version_6/checkpoints/epoch=19-step=2000.ckpt"
autoencoder = model.LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=model.encoder, decoder=model.decoder)
autoencoder = autoencoder.to('cpu')

# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()


# class MyModel(LightningModule):
#     def predict_step(self, batch, batch_idx, dataloader_idx=0):
#         return self(batch)


test_acceleration = []

test_path = "TrainingData/HighPolling/Test"
for directory, subdirectories, files in os.walk(model.path):
    for file in files:
        if "Parsed" in file:
            data = np.genfromtxt(os.path.join(directory, file), dtype=float, delimiter=',', names=True)
            target_array_shape = (45, 4)
            pad_x = (target_array_shape[0]-data.shape[0])

            data = [list(item) for item in data]
            print(data)
            if len(data[0]) == 5:
                print(data)
            data = np.pad(data, ((pad_x, 0), (0,0)), mode="constant")
            print(len(data))

            test_acceleration.append(data)
            test_acceleration = np.array(test_acceleration)
            tensor_acceleration = torch.Tensor(test_acceleration).to('cpu')

            embeddings = encoder(tensor_acceleration)
            print(f'{os.path.join(directory, file)}: {embeddings}')
