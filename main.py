from DataLoader import DataLoader

from neighborhoodBased import kNN

import numpy as np


# Import surprise module
from surprise.prediction_algorithms.knns import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split


file_path = "movielens-sample/rating.csv"
reader = Reader(line_format='user item rating timestamp', sep=",", skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)

trainset, testset = train_test_split(data, test_size=.2)

# Config for surprise similarity function
sim_options = {
    'name': 'cosine',
    'user_based': False
}

algo = KNNBasic(k=10, sim_options=sim_options)
algo.fit(trainset)

predictions = algo.test(testset)
accuracy.rmse(predictions)


print("\nReimlementation of Basic KNN:")
train_data, test_data = DataLoader("movielens-sample", test_ratio=0.2).load()

knn = kNN(data=train_data, k=10, distance="cosine", uuCF=0)
knn.fit()
print (f'RMSE = {knn.rmse(test_data)}')
