import os
import pandas as pd
import numpy as np

from scipy import sparse


class DataLoader:
    """
    Contain methods to read and split dataset into training set and test set.

    Args:
            data_folder (string): Path to folder contain dataset
            test_ratio (float, optional): Size ratio of test set to dataset. Defaults to 0.1.
    """
    def __init__(self, data_folder):
        self.__data_folder = data_folder

        self.__train_data = []
        self.__val_data = []
        self.__test_data = []

    def __read_ratings(self):
        """ Read ratings from data_folder
        """
        names = ['u_id', 'i_id', 'rating', 'timestamp']
        self.__train_data = pd.read_csv(
            self.__data_folder + "/rating_train.csv",
            header=0, names=names
        )
        self.__test_data = pd.read_csv(
            self.__data_folder + "/rating_test.csv",
            header=0, names=names
        )
        self.__val_data = pd.read_csv(
            self.__data_folder + "/rating_val.csv",
            header=0, names=names
        )

    def load_df(self):
        """Load data as DataFrame

        Returns:
            train, val, test
        """
        self.__read_ratings()

        return self.__train_data, self.__val_data, self.__test_data

    def load_sparse(self):
        """Convert dataframe of training set to scipy.sparse matrix
        Row u is the ratings that user u has given to all item.
        Column i is the ratings that all users have given to item i.
        """
        self.__read_ratings()

        train_data = sparse.csr_matrix((
            self.__train_data["rating"],
            (self.__train_data["u_id"], self.__train_data["i_id"])
        ))

        # Reset the index of test_data dataframe to 0
        self.__test_data.reset_index(drop=True, inplace=True)

        return train_data, self.__test_data

    def load_genome(self):
        names = ['i_id', 'g_id', 'score']
        scores = pd.read_csv(
            self.__data_folder + "/genome_scores.csv",
            header=0, names=names
        )

        return scores
