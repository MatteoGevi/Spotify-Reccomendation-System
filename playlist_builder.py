# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
from scipy.sparse import csr_matrix, hstack
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px

# +
'''Playlist Processing for Model Benchmark'''

# Check the Spotify documentation to have some tips regarding how to build your own model
# https://beta.developer.spotify.com/documentation/web-api/reference/browse/get-recommendations/
rec_tracks = []
for i in playlist_df['id'].values.tolist():
    rec_tracks += sp.recommendations(seed_tracks=[i], limit=int(len(playlist_df)/2))['tracks'];

rec_track_ids = []
rec_track_names = []
for i in rec_tracks:
    rec_track_ids.append(i['id'])
    rec_track_names.append(i['name'])

rec_features = []
for i in range(0,len(rec_track_ids)):
    rec_audio_features = sp.audio_features(rec_track_ids[i])
    for track in rec_audio_features:
        rec_features.append(track)
        
rec_playlist_df = pd.DataFrame(rec_features, index = rec_track_ids)


# +
'''Principal Component Analysis for Playlist Building'''

X_train = playlist_df.drop(['id', 'ratings'], axis=1)
y_train = playlist_df['ratings']

X_scaled = StandardScaler().fit_transform(X_train)
pca = decomposition.PCA().fit(X_scaled)
# Fit your dataset to the optimal pca
pca = decomposition.PCA(n_components=8)
X_pca = pca.fit_transform(X_scaled)

# Spotify aso use Word2Vec and Annoy over TF-IDF Vectorizer to suggest new potential songs. So no need to waste time on this

# +
'''K-Neighbours and 5K Cross Validation'''

X_train_last = csr_matrix(hstack([X_pca, X_names_sparse]))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# KNeighbours Classification
knn_params = {'n_neighbors': range(1, 10)}
knn = KNeighborsClassifier(n_jobs=-1)
# GridSearch validation
knn_grid = GridSearchCV(knn, knn_params, cv=skf, n_jobs=-1, verbose=True)
knn_grid.fit(X_train_last, y_train)
knn_grid.best_params_, knn_grid.best_score_

# +
'''Random Forest and Decision Tree'''

parameters = {'max_features': [4, 7, 8, 10], 'min_samples_leaf': [1, 3, 5, 8], 'max_depth': [3, 5, 8]}
rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
forest_grid = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
forest_grid.fit(X_train_last, y_train)
forest_grid.best_estimator_, forest_grid.best_score_

# Decision Tree Classification
tree = DecisionTreeClassifier()
tree_params = {'max_depth': range(1,11), 'max_features': range(4,19)}
tree_grid = GridSearchCV(tree, tree_params, cv=skf, n_jobs=-1, verbose=True)
tree_grid.fit(X_train_last, y_train)
tree_grid.best_estimator_, tree_grid.best_score_

# +
'''XGBoost and Popularity Reccomender'''


# -

'''Content-based vs Collaborative: the Hybrid Reccomender'''



