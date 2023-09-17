from flask import Flask, request
import data
import spotifyHandler
# import model

# To run: python -m flask --app pedalServer run --host=0.0.0.0
app = Flask(__name__)

@app.route('/post', methods=['POST'])
def result():
    csvStr = request.form['csv_as_str']
    with open("./receivedData.csv", "tw", encoding="utf8", newline="") as F:
        F.write(csvStr)
    data.parseFile("./receivedData.csv")
    file = "./receivedData/receivedData-Parsed0.csv"
    # Trigger the AI model then return the result from spotify
    result = 2
    return str(result)