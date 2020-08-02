import os
import pandas as pd


def read_data():
    data_dir = "movielens10k/rating.csv"

    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(data_dir, names=names, engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items


data, num_users, num_items = read_data()
sparsity = 1 - len(data) / (num_users * num_items)
print(f'number of users: {num_users}, number of items: {num_items}')
print(f'matrix sparsity: {sparsity:f}')
print(data.head(5))