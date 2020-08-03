import os
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, data_folder):
        self.__data_folder = data_folder
        self.__ratings = []
        self.__train_data = []
        self.__test_data = []

    def read_data(self):
        """ Read ratings from data_folder
        """
        names = ['user_id', 'item_id', 'rating', 'timestamp']
        ratings = pd.read_csv(self.__data_folder + "/rating.csv", header=0, names=names, engine='python')

        self.__ratings = ratings

    def split_data(self, test_ratio=0.1):
        """Split dataset into training set and test set.

        Args:
            test_ratio (float, optional): Ratio of test set to dataset. Defaults to 0.1.
        """
        mask = [True if x == 1 else False
                for x in np.random.uniform(0, 1, (len(self.__ratings))) < 1 - test_ratio
            ]
        neg_mask = [not x for x in mask]
        train_data, test_data = self.__ratings[mask], self.__ratings[neg_mask]

        return train_data, test_data