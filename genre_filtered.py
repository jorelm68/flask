#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib


# In[3]:
def read_in_data():
    included_variables = ["Popularity", "Danceability","Energy","Loudness","Speechiness","Acousticness","Instrumentalness","Liveness","Tempo","Valence", "Album Release Date"]
    indy_included = included_variables + ["Index"]
    
    df = pd.read_csv("songs_data.csv", sep=',', converters={'Artist Genres': lambda x: x.split(", ")}, keep_default_na=False)
    
    #extract simpler genres
    genres = ['pop', 'rock', 'hip hop', "house", "disco", "soul", "r&b"]
    song_genres = []
    for i in range(0, df.shape[0]):
        row = str(df["Artist Genres"].values[i][0])
        song_genres.append('')
        for g in genres:
            if row.find(g) != -1:
                song_genres[i] = g
                break
    
    df["Genre"] = song_genres
    df_all = pd.read_csv("songs_data.csv")
    df_all["Genre"] = song_genres
    df_all["Index"] = df_all.index

    df_all["Album Release Date"] = pd.to_datetime(df_all["Album Release Date"], format="mixed",  errors='coerce').view('int64') 
    return df_all

# In[8]:
included_variables = ["Popularity", "Danceability","Energy","Loudness","Speechiness","Acousticness","Instrumentalness","Liveness","Tempo","Valence", "Album Release Date"]
indy_included = included_variables + ["Index"]
print(indy_included)

# In[6]:


def clean_and_norm(df_in):
    #normalize the data
    df_ints = pd.DataFrame(df_in, columns = indy_included)
    
    normed=(df_ints-df_ints.mean())/df_ints.std()
    normed=(df_ints-df_ints.min())/(df_ints.max()-df_ints.min())

    #clean data
    for column in indy_included:
        mean = float(normed[column].mean(skipna = True))
        normed[column] = normed[column].replace(np.nan, mean)
    return normed


# In[7]:


def knn_fit(name, data):
    #clean data
    normed = clean_and_norm(data)

    #grab normed data with only included variables
    train_data = pd.DataFrame(normed, columns = included_variables)
    train_data = train_data.iloc[1:,]
    
    knn = NearestNeighbors(metric='euclidean', algorithm='auto')
    knn.fit(train_data)

    song_i = int(data["Index"].iloc[0]) 
    test = pd.DataFrame(data, columns = included_variables)
    test = test.iloc[song_i:song_i+1:,]

    distances, indices = knn.kneighbors(test, n_neighbors=3)
    
    '''print(data["Track Name"].iloc[song_i], "by", data["Artist Name(s)"].iloc[song_i])
    print()
    print("Closest to:")
    
    for ind in indices[0]:
        print(data["Track Name"].iloc[ind + 1], "by", data["Artist Name(s)"].iloc[ind + 1])
    print()'''

    # save that model uh huh uh huh
    # load via: https://datascience.stackexchange.com/questions/52704/how-to-save-a-knn-model
    # joblib.dump(knn, name)


# In[14]:


def move_knn(data, query, variable, direction):
    #clean data
    normed = clean_and_norm(data)

    #grab normed data with only included variables
    train_data = pd.DataFrame(normed, columns = included_variables)
    
    knn = NearestNeighbors(metric='euclidean', algorithm='auto')
    knn.fit(train_data)

    query_stripped = pd.DataFrame(normed, columns = included_variables)
    distances, indices = knn.kneighbors(query_stripped, n_neighbors=3)

    print("Input Song:")
    print(query["Track Name"], "by", query["Artist Name(s)"])

    # where are we going good sir
    direction_str = "more"
    if direction == 0: direction_str = "less"

    print("Our output has", direction_str, variable, "than our input:")
    ind = indices[0][1]
    print(data["Track Name"].iloc[ind + 1], "by", data["Artist Name(s)"].iloc[ind + 1])
    print(ind)
    print()

    return ind


# In[12]:


def move_along(data, query_point, variable, direction):
    '''Data = input data (should be genre specific)
    query_point = current song
    variable = what are we changing? Acousticness? Loudness?
    direction = up/down (1 = up, 0 = down)'''
    filtered_data = data[data[variable].iloc[:,] > query_point[variable]]
    if direction == 0:    
        filtered_data = data[data[variable].iloc[:,] < query_point[variable]]
    
    print(filtered_data.head())

    # Ensure there is at least one point left after filtering
    if len(filtered_data) == 0:
        print("No neighbors found with a higher", variable,"value than the query point.")
        return query_point["Index"]
    else:
        return move_knn(filtered_data, query_point, variable, direction)

# In[ ]:
'''
genres = ['pop', 'rock', 'hip hop', "house", "disco", "soul", "r&b"]
for genre in genres:
    print(genre)
    g_data = df_all[df_all['Genre'] == genre]
    name = genre + ".pkl"
    knn_fit(name, g_data)
    print("------------------------")
'''
# In[ ]:

def run_ml_back(query_index, variable, direction):
    df_all = read_in_data()

    query = df_all.iloc[int(query_index)]

    genre = query["Genre"]

    g_data = df_all[df_all['Genre'] == genre]

    return move_along(g_data, query, variable, int(direction))

# In[ ]:
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import matplotlib.pyplot as plt

# In[ ]:
def graph_out(index):
    df_all = read_in_data()
    
    query = df_all.iloc[int(index)]

    genre = query["Genre"]

    g_data = df_all[df_all['Genre'] == genre]

    return knn_graph(g_data, index)

def knn_graph(data, index):
    indy_data = pd.DataFrame(data, columns = indy_included)
    #clean data
    normed = clean_and_norm(data)

    #grab normed data with only included variables
    train_data = pd.DataFrame(normed, columns = included_variables)
    
    knn = NearestNeighbors(metric='euclidean', algorithm='auto', n_neighbors=5)
    knn.fit(train_data)

    distances, indices = knn.kneighbors(train_data, n_neighbors=3)

    # graphing children hehe
    G = nx.Graph()
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            G.add_edge(i, neighbor, weight=distances[i][np.where(neighbors == neighbor)[0][0]])

    
    start = index
    end = 2

    # Find the shortest path using Dijkstra's algorithm
    try:
        path = nx.shortest_path(G, source=start, target=end, weight='weight')
        print("Shortest path from {} to {} is: {}".format(start, end, path))
        print("Path coordinates:", [indy_data["Index"].iloc[i] for i in path])
        path_to_return = []
        for i, ind in enumerate(path):
            print(int(indy_data["Index"].iloc[ind]))
            path_to_return.append(data[indy_data["Index"].iloc[ind]])
            print(data["Track Name"].iloc[ind], "by", data["Artist Name(s)"].iloc[ind])
        
        return path_to_return
        # visualize_tree(train_data, G, path, start)
    except nx.NetworkXNoPath:
        print("No path found between {} and {}".format(start, end))
