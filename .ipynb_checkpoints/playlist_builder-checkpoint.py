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
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px

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
# -

'''Random Forest and Decision Tree'''

