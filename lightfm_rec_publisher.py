from lightfm.data import Dataset
from lightfm import LightFM
from dataset import BookRatingDataset, FantasyUser, RomanceUser, ScifiUser
import pandas as pd


def build_dataset(ds, rating_threshold=6):
    """Build LightFM dataset from pandas Book crossing dataset"""

    # create unique embedding for each unique user/item
    lfm_ds = Dataset(user_identity_features=True, item_identity_features=True)

    # users - iterable of user ids
    # items - iterable of item ids
    lfm_ds.fit(users=ds.users.user_id, items=ds.books.ISBN)
    lfm_ds.fit_partial(items=(isbn for isbn in ds.books.ISBN), item_features=(f"author:{author}" for author in ds.books.book_author))
    lfm_ds.fit_partial(items=(isbn for isbn in ds.books.ISBN), item_features=(f"publisher:{publisher}" for publisher in ds.books.publisher))

    # data - iterable of (user_id, item_id, weight)
    interactions, weights = lfm_ds.build_interactions(data=(((row.user_id, row.ISBN) for index, row in ds.ratings.iterrows() if row.book_rating > rating_threshold)))

    item_features = lfm_ds.build_item_features(((row.ISBN, [f"author:{row.book_author}", f"publisher:{row.publisher}"]) for index, row in ds.books.iterrows()))

    return lfm_ds, interactions, item_features


def recommend(uid, item_features, n=30):
    uid_to_id = lfm_ds.mapping()[0]
    isbn_to_id = lfm_ds.mapping()[2]

    pred_ratings = pd.DataFrame({
        "ISBN":
        ds.books.ISBN,
        "pred_rating":
        model.predict(uid_to_id[uid], ds.books.ISBN.apply(lambda isbn: isbn_to_id[isbn]).values, item_features=item_features)
    })

    return pred_ratings.sort_values("pred_rating", ascending=False)[:n].merge(ds.books, on="ISBN")


pd.set_option("max_colwidth", 100)
ds = BookRatingDataset(implicit=False)

fantasy_user = FantasyUser(ds)
scifi_user = ScifiUser(ds)
romance_user = RomanceUser(ds)

lfm_ds, interactions, item_features = build_dataset(ds)
num_users, num_items = lfm_ds.interactions_shape()
print(f'Num users: {num_users}, num_items {num_items}.')
print(f"Item features shape: {lfm_ds.item_features_shape()}")


alpha = 1e-05
epochs = 300
num_components = 30

model = LightFM(loss='warp', user_alpha=alpha, item_alpha=alpha, no_components=num_components)

model.fit(interactions=interactions, item_features=item_features, epochs=epochs, num_threads=4, verbose=True)


recommend(fantasy_user.uid, item_features)
recommend(scifi_user.uid, item_features)
recommend(romance_user.uid, item_features)
