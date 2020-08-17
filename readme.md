# Recommender System on MovieLens 20M Dataset (working in progress)

## Dataset
In this repo, I'm using the Movielens 20M Dataset from [Kaggle](https://www.kaggle.com/grouplens/movielens-20m-dataset).

This is a big dataset.
In order to extract the dataset to get a smaller dataset, first you need to download MovieLens 20M and save it on your computer, for example, to `movielens20M` folder.
Then you need to create a folder `movilens-sample` for the new sampling dataset.

On `sample_movielens.py` you can change the parameter to your like.

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

## Run the Algorithm
`main.py` is where all the magic happens.

```python
train_data, test_data = DataLoader("movielens-sample", test_ratio=0.2).load()
```

Update the folder name where you put the sampling data (or the whole MovieLens 20M dataset), change `test_ratio` (default ratio is 0.1).

Then on your terminal, run
```
py main.py
```

After a while, final RMSE score will be display to the terminal, and also the runtime.
