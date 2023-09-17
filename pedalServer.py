from flask import Flask, request
import os
import data
import spotifyHandler
import torch
import numpy as np

from model import SimpleNN

loaded_model = torch.load('trained_model.pth')
loaded_model.eval()
emotionsMap = {"happy": 0, "sad": 1, "chill": 2, "angry": 3, "invalid": 4}


def predict(file):
    input_data = np.genfromtxt(file, dtype=float, delimiter=',', names=True)
    print("hi")
    print(f'data: {input_data}')
    target_array_shape = (45, 3)
    pad_x = (target_array_shape[0] - input_data.shape[0])

    input_data = [list(item) for item in input_data]
    input_data = np.delete(input_data, 3, 1)
    input_data = np.pad(input_data, ((pad_x, 0), (0, 0)), mode="constant")

    input_data = input_data.flatten()

    test_data_array = input_data

    test_data_array = np.vstack((test_data_array, np.zeros((135,))))

    test_data_array = np.array(test_data_array, dtype="float32")
    print(type(test_data_array[0][0]))
    test_data_array = torch.Tensor(test_data_array)

    with torch.no_grad():
        predicted_outputs = loaded_model(test_data_array)
        _, predicted_labels = torch.max(predicted_outputs, 1)
        predicted_labels = predicted_labels.tolist()

        return predicted_labels[0]


# To run: python -m flask --app pedalServer run --host=0.0.0.0
app = Flask(__name__)

@app.route('/post', methods=['POST'])
def result():
    try:
        print(request.form, request.headers)
        csvStr = request.form['csv_as_str']
        with open("./receivedData.csv", "tw", encoding="utf8", newline="") as F:
            F.write(csvStr)
        data.parseFile("./receivedData.csv")
        fileCount = len([file for file in os.listdir("./receivedData/") if os.path.isfile(os.path.join("./receivedData/", file))])
        file = f"./receivedData/receivedData-Parsed{fileCount-1}.csv" # Take the last 2-second entry
        # Trigger the AI model then return the result from spotify
        result = predict(file)
        for f in os.listdir("./receivedData/"):
            os.remove(os.path.join("./receivedData/", f))
        os.rmdir("./receivedData/")
        return str(result) + "\n" + "\n".join([item for item in spotifyHandler.playlistGeneration(spotifyHandler.getGenres(result), 5)])
    except Exception as e:
        print(e)
        return ""