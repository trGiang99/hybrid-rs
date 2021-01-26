import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

from utils import DataLoader
# from svd_knn import hybrid
from svd import svdpp


print("Loading data ...")
loader = DataLoader(
    data_folder="movielens10k",
    genome_folder="movielens10k"
)
train_data, test_data = loader.load_csv2df(use_val=False)
movie_genome = loader.load_genome_fromcsv()

# knn_options = {
#     'k': 20,
#     'distance': 'cosine',
# }

# nfactors = movie_genome.shape[1]
# svd_options = {
#     'learning_rate': 0.001,
#     'regularization': 0.02,
#     'n_epochs': 20,
#     'n_factors': nfactors,
#     'min_rating': 0.5,
#     'max_rating': 5
# }

# model = hybrid(knn_options=knn_options, svd_options=svd_options)
# model.fit(train_data=train_data, movie_genome=movie_genome)

nfactors = 1128
model = svdpp(
    learning_rate=0.005,
    regularization=0.02,
    n_epochs=20, n_factors=nfactors,
    min_rating=0.5, max_rating=5
)
model.fit(X=train_data, i_factor=movie_genome)

pred = model.predict(test_data)
rmse = sqrt(mean_squared_error(test_data["rating"], pred))
mae = mean_absolute_error(test_data['rating'], pred)

print(f'\nTest RMSE: {rmse:.5f}')
print(f'Test MAE: {mae:.5f}')

# model.save_checkpoint("checkpoint.pkl")
