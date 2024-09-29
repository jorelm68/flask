from flask import Flask, request, jsonify
from genre_filtered import run_ml_back

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib
import os

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
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)