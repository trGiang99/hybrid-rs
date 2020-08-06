import os
import pandas as pd
import numpy as np

from scipy import sparse


class DataLoader:
    """
    Contain methods to read and split dataset into training set and test set.
    """
    def __init__(self, data_folder):
        """
        Args:
            data_folder (string): Path to folder contain dataset
        """
        self.__data_folder = data_folder
        self.__ratings = []
        self.__train_data = []
        self.__test_data = []

    def read_data(self):
        """ Read ratings from data_folder
        """
        names = ['user_id', 'item_id', 'rating', 'timestamp']
        self.__ratings = pd.read_csv(
            self.__data_folder + "/rating.csv",
            header=0, names=names
        )

    def split_data(self, test_ratio=0.1):
        """Randomly split dataset into training set and test set.

        Args:
            test_ratio (float, optional): Size ratio of test set to dataset. Defaults to 0.1.
        """
        mask = [
            True if x == 1 else False
                for x in np.random.uniform(0, 1, (len(self.__ratings))) < 1 - test_ratio
        ]

        # Keep the last user and item on the training set
        #   to maintain the shape of the utility matrix.
        mask[-1] = True
        mask[self.__ratings["item_id"].idxmax(axis=0)] = True

        neg_mask = [not x for x in mask]
        self.__train_data, self.__test_data = self.__ratings[mask], self.__ratings[neg_mask]

    def load_data(self):
        """Convert dataframe of training set to scipy.sparse matrix
        Row u is the ratings that user u has given to all item.
        Column i is the ratings that all users have given to item i.
        """
        train_data = sparse.csr_matrix((
            self.__train_data["rating"],
            (self.__train_data["user_id"], self.__train_data["item_id"])
        ))

        return train_data, self.__test_data