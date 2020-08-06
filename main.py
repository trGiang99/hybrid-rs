from DataLoader import DataLoader

from neighborhoodBased import kNN

import numpy as np

train_data, test_data = DataLoader("movielens-sample", test_ratio=0.2).load()

# print (f"Training Data:\n{train_data[1, 3120]}")
# print (f"Test Data:\n{test_data['user_id'][2000]}")
# exit()


knn = kNN(data=train_data, k=10, distance="cosine", uuCF=0)


squared_error = 0
n_user = test_data.shape[0]

for n in range(n_user):
    print(f"Predicting {test_data['user_id'][n]},{test_data['item_id'][n]}")
    pred = knn.prediction(test_data["user_id"][n], test_data["item_id"][n])
    squared_error += (pred - test_data["rating"][n])**2

RMSE = np.sqrt(squared_error/n_user)
print (f'User-user CF, RMSE = {RMSE}')
