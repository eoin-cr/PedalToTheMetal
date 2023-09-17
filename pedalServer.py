from flask import Flask, request
import os
import data
import spotifyHandler
# import model

# To run: python -m flask --app pedalServer run --host=0.0.0.0
app = Flask(__name__)

@app.route('/post', methods=['POST'])
def result():
    try:
        print(request.args, request.headers)
        csvStr = request.form['csv_as_str']
        with open("./receivedData.csv", "tw", encoding="utf8", newline="") as F:
            F.write(csvStr)
        data.parseFile("./receivedData.csv")
        fileCount = len([file for file in os.listdir("./receivedData/") if os.path.isfile(os.path.join("./receivedData/", file))])
        file = f"./receivedData/receivedData-Parsed{fileCount-1}.csv" # Take the last 2-second entry
        # Trigger the AI model then return the result from spotify
        result = 2
        for f in os.listdir("./receivedData/"):
            os.remove(f)
        os.rmdir("./receivedData/")
        return str(result)
    except Exception as e:
        print(e)
        return ""