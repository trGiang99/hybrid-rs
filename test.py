import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


from utils import DataLoader
from knn import kNN
from svd import svd


def test_knn():
    loader = DataLoader(
        data_folder="movielens10k",
        genome_folder="movielens10k"
    )

    train_data, test_data = loader.load_sparse()
    movie_genome = loader.load_genome_fromcsv()

    knn = kNN(k=5, distance="cosine", uuCF=0, normalize="mean")
    knn.fit(train_data=train_data, genome=movie_genome)
    print (f'\nRMSE: {knn.rmse(test_data)}')


def test_svd():
    nfactors = 1128

    model = svd(
        learning_rate=0.005,
        regularization=0.02,
        n_epochs=10, n_factors=nfactors,
        min_rating=0.5, max_rating=5
    )

    loader = DataLoader(
        data_folder="movielens10k",
        genome_folder="movielens10k"
    )

    train, test = loader.load_csv2df(use_val=False)
    movie_genome = loader.load_genome_fromcsv()

    model.fit(
        X=train,
        # X_val=val,
        i_factor=movie_genome,
        # u_factor=user_genome,
        early_stopping=False, shuffle=False
    )

    pred = model.predict(test)
    rmse = sqrt(mean_squared_error(test["rating"], pred))
    mae = mean_absolute_error(test['rating'], pred)

    print(f'\nTest RMSE: {rmse:.5f}')
    print(f'Test MAE: {mae:.5f}')


if __name__ == "__main__":
    # test_knn()
    test_svd()
