from abc import abstractmethod
from dataset import BookRatingDataset, FantasyUser
import pandas as pd
from surprise import Reader, Dataset, NormalPredictor, SVD
from surprise.model_selection import cross_validate
from sklearn import preprocessing
import random

pd.set_option("max_colwidth", 100)

ds = BookRatingDataset()
fantasy_user = FantasyUser(ds)

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ds.ratings, reader)

trainset = data.build_full_trainset()

# If biased==True, the user ratings would be normalized and he would need to also supply some negative rating information as well
algo = SVD(biased=True, n_factors=20)

print("\n\nFitting SVD...")
algo.fit(trainset)

print("Generating recommendations")
fantasy_user.recommend_books(algo, threshold=30)

# print("\n\nFitting NormalPredictor...")
# cross_validate(NormalPredictor(), data, cv=2, verbose=True)

# cross_validate(SVD(), data, cv=2, verbose=True)
