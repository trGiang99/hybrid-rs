import os
from DataLoader import DataLoader

manipulator = DataLoader("movielens-sample")

manipulator.read_data()

train_data, test_data = manipulator.split_data(test_ratio=0.2)

print (f"Training Data:\nLenght: {len(train_data)}")
print(train_data.head(5))

print (f"Test Data:\nLenght: {len(test_data)}")
print(test_data.head(5))