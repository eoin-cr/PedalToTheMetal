import os

import torch
import numpy as np

from model import SimpleNN

loaded_model = torch.load('trained_model.pth')

loaded_model.eval()

test_path = "TrainingData/HighPolling/Test3/"
emotionsMap = {"happy": 0, "sad": 1, "chill": 2, "angry": 3, "invalid": 4}


test_data_array = None
filesList = []

for directory, subdirectories, files in os.walk(test_path):
    for file in files:
        print(file)
        if "Parsed" in file:
            os_path = os.path.join(directory, file)
            filesList.append(os_path)
            input_data = np.genfromtxt(os_path, dtype=float, delimiter=',', names=True)
            print("hi")
            print(f'data: {input_data}')
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

print(test_data_array.shape)

test_data_array = np.vstack((test_data_array, np.zeros((135,))))

test_data_array = np.array(test_data_array, dtype="float32")
print(type(test_data_array[0][0]))
test_data_array = torch.Tensor(test_data_array)

i = 0

correct = []

for file in filesList:
    for key in emotionsMap.keys():
        if key in file:
            correct.append(emotionsMap[key])
            break

with torch.no_grad():
    predicted_outputs = loaded_model(test_data_array)
    _, predicted_labels = torch.max(predicted_outputs, 1)
    predicted_labels = predicted_labels.tolist()
    predicted_labels.pop(-1)

    print(predicted_labels)
    print(correct)
