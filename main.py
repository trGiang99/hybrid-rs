from DataLoader import DataLoader

from neighborhoodBased import kNN

import numpy as np


# Import surprise module
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.model_selection import cross_validate
from surprise import Dataset
from surprise import Reader


file_path = "movielens-sample/rating.csv"
reader = Reader(line_format='user item rating timestamp', sep=",", skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)

# Config for surprise similarity function
sim_options = {
    'name': 'cosine',
    'user_based': False
}

algo = KNNBasic(k=10, sim_options=sim_options)

print("Surprise Module:")
cross_validate(algo, data, verbose=True)


print("\nBasic KNN:")
train_data, test_data = DataLoader("movielens-sample", test_ratio=0.2).load()

# print (f"Training Data:\n{train_data[1, 3120]}")
# print (f"Test Data:\n{test_data['user_id'][2000]}")
# exit()


knn = kNN(data=train_data, k=10, distance="cosine", uuCF=0)


squared_error = 0
n_test_ratings = test_data.shape[0]

for n in range(n_test_ratings):
    # print(f"Predicting {test_data['user_id'][n]},{test_data['item_id'][n]}")
    pred = knn.prediction(test_data["user_id"][n], test_data["item_id"][n])
    squared_error += (pred - test_data["rating"][n])**2

RMSE = np.sqrt(squared_error/n_test_ratings)
print (f'User-user CF, RMSE = {RMSE}')
