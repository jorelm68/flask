from flask import Flask, request, jsonify
from genre_filtered import run_ml_back

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "Hello, World!"

@app.route('/api/run_ml', methods=['POST'])
def run_ml():
    data = request.json
    index = data.get('index')
    variable = data.get('variable')
    direction = data.get('direction')

    result = run_ml_back(index, variable, direction)

    # Your ML logic here
    # For now, we'll just return the input data
    res = {
        "index": index,
        "variable": variable,
        "direction": direction,
        "result": int(result)
    }
    return jsonify(res)

# This is important for Vercel deployment
if __name__ == '__main__':
    app.run()