import pandas as pd
import numpy as np

from scipy import sparse


class DataLoader:
    """
    Contain methods to read and split dataset into training set and test set.

    Args:
            data_folder (string): Path to folder that contain dataset
            genome_folder (string): Path to folder that contain the genome_scores file
    """
    def __init__(self, data_folder, genome_folder=None):
        self.__data_folder = data_folder
        if genome_folder is None:
            self.__genome_folder = data_folder
        else:
            self.__genome_folder = genome_folder

        self.__train_data = None
        self.__val_data = None
        self.__test_data = None

    def __read_trainset(self, columns):
        self.__train_data = pd.read_csv(
            self.__data_folder + "/rating_train.csv",
            header=0, names=columns
        )

    def __read_testset(self, columns):
        self.__test_data = pd.read_csv(
            self.__data_folder + "/rating_test.csv",
            header=0, names=columns
        )

    def __read_valset(self, columns):
        self.__val_data = pd.read_csv(
            self.__data_folder + "/rating_val.csv",
            header=0, names=columns
        )

    def load_csv2df(self, use_val=True, columns=['u_id', 'i_id', 'rating', 'timestamp']):
        """
        Load training set, validate set and test set.
        Each as DataFrame

        Args:
            has_val (boolean): Denote if using validate data or not. Defaults to True.
            columns (list): Columns name for DataFrame. Defaults to ['u_id', 'i_id', 'rating', 'timestamp'].

        Returns:
            train, val, test (DataFrame)
        """
        self.__read_trainset(columns)
        self.__read_testset(columns)

        if use_val:
            self.__read_valset(columns)

        return self.__train_data, self.__val_data, self.__test_data

    def load_genome_fromcsv(self, genome_file="genome_scores.csv", columns=["i_id", "g_id", "score"], reset_index=False):
        """
        Load genome scores from file.
        Args:
            genome_file (string): File name that contain genome scores. Must be in csv format.
            columns (list, optional): Columns name for DataFrame. Must be ["i_id", "g_id", "score"] or ["i_id", "score", "g_id"].
            reset_index (boolean): Reset the genome_tag column or not. Defaults to False.

        Returns:
            scores (DataFrame)
        """
        scores = pd.read_csv(
            self.__genome_folder + "/" + genome_file,
            header=0, names=columns
        )

        if reset_index:
            tag_map = {genome.g_id: (newIdx+1) for newIdx, genome in scores.loc[scores.i_id == 1].iterrows()}
            scores["g_id"] = scores["g_id"].map(tag_map)

        return scores

    def load_sparse(self):
        """Convert dataframe of training set to scipy.sparse matrix
        Row u is the ratings that user u has given to all item.
        Column i is the ratings that all users have given to item i.
        """
        columns = ['u_id', 'i_id', 'rating', 'timestamp']

        self.__read_trainset(columns)
        self.__read_testset(columns)

        train_data = sparse.csr_matrix((
            self.__train_data["rating"],
            (self.__train_data["u_id"], self.__train_data["i_id"])
        ))

        # Reset the index of test_data dataframe to 0
        self.__test_data.reset_index(drop=True, inplace=True)

        return train_data, self.__test_data
