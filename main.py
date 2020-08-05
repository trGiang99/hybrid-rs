from DataLoader import DataLoader

from neighborhoodBased import kNN

manipulator = DataLoader("movielens-sample")

manipulator.read_data()

manipulator.split_data(test_ratio=0.2)

train_data, test_data = manipulator.load_data()


# print (f"Training Data:\n{train_data}")
# print (f"Test Data:\n{test_data}")


knn = kNN(data=train_data, k=10, distance="cosine", uuCF=0)

print(knn.prediction(1, 11))