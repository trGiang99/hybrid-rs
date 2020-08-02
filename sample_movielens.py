# TODO: add 2 more function to read - extract all necessary genome tags and write to the new path


import csv


def sample_movielens(movielens_path, movielens_sample_path, sample_size=10000):
    """Sampling the original MovieLens Dataset into a dataset contain only 10k row
    All movies that doesn't appear in the raing file will be remove from the original movies.csv
    Also the genome_tags and genome_score.

    Args:
        movielens_path (string): Folder contains the original MovieLens Dataset
        movielens_sample_path (string): Folder to save new 10k MovieLens Dataset
    """

    # List contains all movie's IDs in the new dataset
    movies_list = []

    with open(movielens_path + "/rating.csv", 'r') as movielens:
        reader = csv.reader(movielens)

        rating_list = []

        for idx, rating in enumerate(reader):
            rating_list.append(rating)
            if rating[1] not in movies_list:
                movies_list.append(rating[1])          # Append new movie's Id to the movie list

            # Copy only first 10,000 (sample_size) rows
            if (idx > sample_size):
                break

    genome_scores = []
    # Get all genome score for the Ids in the movie_list above
    with open(movielens_path + "/genome_scores.csv", "r", encoding="utf-8") as scores:
        reader = csv.reader(scores)
        genome_scores = [score for _, score in enumerate(reader) if score[0] in movies_list]

    # Get information of all movies and assign it to the Ids in movie_list respectively
    with open(movielens_path + "/movie.csv", 'r', encoding="utf-8") as movies:
        reader = csv.reader(movies)
        movies_list = [movie for _, movie in enumerate(reader) if movie[0] in movies_list]

    # Write 10k ratings to new path
    with open(movielens_sample_path + "/rating.csv", 'w') as ratings:
        writer = csv.writer(ratings)
        writer.writerows(rating_list)

    # Write all movies in the movie list to new path
    with open(movielens_sample_path + "/movie.csv", 'w', encoding="utf-8") as movies:
        writer = csv.writer(movies)
        writer.writerows(movies_list)

    # Write all needed score to new path
    with open(movielens_sample_path + "/genome_scores.csv", 'w', encoding="utf-8") as scores:
        writer = csv.writer(scores)
        writer.writerows(genome_scores)


if __name__ == "__main__":

    # Sampling MovieLens 20M Dataset to MovieLens 10k Dataset for the sake of testing
    sample_movielens("movielens20M", "movielens-sample")