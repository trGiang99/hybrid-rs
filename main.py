import numpy as np

from loader import DataLoader
from svd_knn import kNN


print("\nReimlementation of KNN with mean normalization:")

train_data, test_data = DataLoader(data_folder="movielens-sample").load_csv2df(use_val=False)

knn = kNN(k=5, distance="cosine", uuCF=1, normalize="mean")
knn.fit(train_data=train_data)
print (f'\nRMSE: {knn.rmse(test_data)}')


print("\nKNN with mean normalization from NicolasHug/Surprise:")
# Import surprise module
from surprise.prediction_algorithms.knns import KNNWithMeans
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
    'user_based': True
}

algo = KNNWithMeans(k=5, sim_options=sim_options)
algo.fit(trainset)

predictions = algo.test(testset)
accuracy.rmse(predictions)
