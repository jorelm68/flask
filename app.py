from flask import Flask, request, jsonify
from genre_filtered import run_ml_back
from genre_filtered import graph_out
from genre_filtered import hint

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib
import os
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import matplotlib.pyplot as plt

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

    df = pd.read_csv("songs_data.csv")

    res_query = df.iloc[int(result)]
    uri = res_query["Track URI"][14:]

    # Your ML logic here
    # For now, we'll just return the input data
    res = {
        "result": int(result),
        "track_id": uri
    }
    return jsonify(res)

@app.route('/shortest_steps', methods=['POST'])
def run_graph():
    data = request.json
    start = data.get('start')
    end = data.get('end')

    df = pd.read_csv("songs_data.csv")

    steps = graph_out(start, end)

    # Your ML logic here
    # For now, we'll just return the input data
    res = {
        "shortest_steps": steps
    }
    return jsonify(res)

@app.route('/ml_random', methods=['POST'])
def random_track():
    # return track id and index
    rand_index = np.random.randint(10000)
    
    df = pd.read_csv("songs_data.csv")

    query = df.iloc[rand_index]
    uri = query["Track URI"][14:]
    res = {
        "index": int(rand_index),
        "track_id": uri
    }
    return jsonify(res)

@app.route('/index_to_uri', methods=['POST'])
def index_to_uri():
    # return track id and index
    data = request.json
    index = data.get('index')
    
    df = pd.read_csv("songs_data.csv")

    query = df.iloc[index]
    uri = query["Track URI"][14:]
    res = {
        "track_id": uri
    }
    return jsonify(res)


@app.route('/index_to_row', methods=['POST'])
def index_to_row():
    # return track id and index
    data = request.json
    index = data.get('index')
    
    df = pd.read_csv("songs_data.csv", dtype=str)

    query = df.iloc[index]

    print(query)

    res = {
        "index": str(index),
        "name": query["Track Name"],
        "artist": query["Artist Name(s)"],
        "album": query["Album Name"],
        "image": query["Album Image URL"],
        "preview": query["Track Preview URL"],
        "Popularity": query["Popularity"],
        "Danceability": query["Danceability"],
        "Energy": query["Energy"],
        "Loudness": query["Loudness"],
        "Speechiness": query["Speechiness"],
        "Acousticness": query["Acousticness"],
        "Instrumentalness": query["Instrumentalness"],
        "Liveness": query["Liveness"],
        "Tempo": query["Tempo"],
        "Valence": query["Valence"],
        "AlbumReleaseDate": query["Album Release Date"]
    }
    
    return jsonify(res)


@app.route('/hint', methods=['POST'])
def hint_route():
    data = request.json
    start = data.get('start')
    end = data.get('end')

    var, dir = hint(start, end)

    # Your ML logic here
    # For now, we'll just return the input data
    res = {
        "variable": var,
        "direction": dir
    }
    return jsonify(res)


# This is important for Vercel deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)