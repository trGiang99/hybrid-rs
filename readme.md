# Recommendation System on MovieLens 20M Dataset (working in progress)

## Dataset
In this repo, I'm using the Movielens 20M Dataset from [Kaggle](https://www.kaggle.com/grouplens/movielens-20m-dataset).

This is a big dataset.
In order to extract the dataset to get a smaller dataset, first you need to download MovieLens 20M and save it on your computer.
Then you need to create a folder for the new sampling dataset.
On `sample_movielens.py` you can change

```python
if __name__ == "__main__":
    sample_movielens(
       "movielens20M",
       "movielens-sample",
       sample_size=1000
    )
```

where `"movielens20M"` is the folder contains MovieLens 20M Dataset, `"movielens-sample"` is the folder contains new extracted dataset.
Size of the extracted dataset can be changed via `sample_size`.
