import numpy as np
from abc import abstractmethod
import pandas as pd
from IPython.display import display


class BookRatingDataset:
    def __init__(self,
                 implicit=False,
                 drop_unknown_ISBN_ratings=True,
                 books_path="dataset/books.csv",
                 users_path="dataset/users.csv",
                 ratings_path="dataset/ratings.csv"):

        self.implicit = implicit
        self.drop_unknown_ISBN_ratings = drop_unknown_ISBN_ratings
        self.books_path = books_path
        self.users_path = users_path
        self.ratings_path = ratings_path

        self.load_dataset()

    def load_dataset(self):
        """Extract data for books, ratings and users from csv files, join on ISBN and user ID"""

        # the .csv file isn't valid, books contain few records with more columns
        # we skip them for simplicity
        print("Loading users...")
        self.users = pd.read_csv(self.users_path, delimiter=";", on_bad_lines="warn")

        print("Loading books...")
        self.books = pd.read_csv(self.books_path, delimiter=";", on_bad_lines="warn")

        print("Loading ratings...")
        self.ratings = pd.read_csv(self.ratings_path, delimiter=";", on_bad_lines="warn")

        print(f"Users dataset rows: {len(self.users)}")
        print(f"Books dataset rows: {len(self.books)}")
        print(f"Ratings dataset rows: {len(self.ratings)}")

        self._clean_users()
        self._clean_ratings()
        self._clean_books()

        if self.drop_unknown_ISBN_ratings:
            ratings_clean = self.ratings.merge(self.books, on="ISBN")[["user_id", "ISBN", "book_rating"]]
            print(f"Dropping {len(self.ratings) - len(ratings_clean)} ratings with unknown book ISBN value")
            self.ratings = ratings_clean

        self.books = self.compute_book_number_of_ratings()
        self.books = self.books[self.books.n_ratings > 0]

    def _clean_users(self):
        # modifies users in-place

        ## Age
        self.users.loc[(self.users.age > 90) | (self.users.age < 5), 'age'] = np.nan
        self.users.age = self.users.age.fillna(self.users.age.median()).astype(np.int32)

        # And maybe we also want user-id to be a int32
        self.users.user_id = self.users.user_id.astype(np.int32)

        ## Location
        loc_segments = pd.DataFrame(self.users.location.str.split(",", n=0, expand=True))

        self.users["city"] = loc_segments[0].str.strip()
        self.users["state"] = loc_segments[1].str.strip()
        self.users["country"] = loc_segments[2].str.strip()

        self.users.drop("location", inplace=True, axis=1)
        self.users.head()

        # Location field contains many mistakes, we fix some
        self.users.loc[self.users.country.str.len() <= 1, ["country"]] = "n/a"
        self.users.loc[self.users.state.str.len() <= 1, ["state"]] = "n/a"
        self.users.loc[self.users.city.str.len() <= 1, ["city"]] = "n/a"
        self.users.fillna("n/a", inplace=True)

    def _clean_books(self):
        # modifies books in-place

        self.books.drop(["Image-URL-S", "Image-URL-M", "Image-URL-L"], axis=1, inplace=True)
        self.books[self.books.year_of_publication == "DK Publishing Inc"]
        self.books[self.books.year_of_publication == "Gallimard"]

        # let's just drop those
        self.books = self.books[self.books.year_of_publication != "DK Publishing Inc"]
        self.books = self.books[self.books.year_of_publication != "Gallimard"]

        # and unify the int and string representations
        self.books.year_of_publication = pd.to_numeric(self.books.year_of_publication)

        # replace the invalid years with a mean
        # have to do this in 2 steps, such that the median isn't biased with the self.books with invalid publication years
        self.books.loc[(self.books.year_of_publication > 2021) | (self.books.year_of_publication < 1500), "year_of_publication"] = np.nan
        self.books.year_of_publication = self.books.year_of_publication.fillna(round(self.books.year_of_publication.median())).astype(np.int32)

        # Numerical attribute year_of_publication has been dealt with, so we can replace all remaining nulls with "n/a"
        self.books.fillna("n/a", inplace=True)

        # we must not forget to convert the ratings ISBNs to lowercase as well!
        self.books.ISBN = self.books.ISBN.str.lower()
        self.books.book_title = self.books.book_title.str.lower()
        self.books.book_author = self.books.book_author.str.lower()
        self.books.publisher = self.books.publisher.str.lower()

        # Let's drop ISBN duplicities
        self.books.drop_duplicates(["ISBN"], inplace=True)

    def _clean_ratings(self):
        # modifies ratings in place
        self.ratings.ISBN = self.ratings.ISBN.str.lower()

        if self.implicit:
            print("Keeping implicit ratings.")
        else:
            print("Loading only explicit ratings (1-10)")
            # drop implicit ratings
            self.ratings = self.ratings[self.ratings.book_rating != 0]

        # drop ratings with unknown users
        num_ratings = len(self.ratings)
        self.ratings = self.ratings.merge(self.users[["user_id"]], on="user_id")
        print(f"Dropping {num_ratings - len(self.ratings)} with unknown user_ids")

    # def find_books_by_title(self, title):
    # self.books[]

    def compute_book_number_of_ratings(self):
        books_n_rated = self.books.merge(self.ratings, on="ISBN").groupby("ISBN").size().reset_index(name="n_ratings")
        books_with_ratings = self.books.merge(books_n_rated, on="ISBN", how="left")
        books_with_ratings.n_ratings.fillna(0, inplace=True)
        books_with_ratings.n_ratings = books_with_ratings.n_ratings.astype(np.int32)
        return books_with_ratings

    def find_books(self, title=None, author=None):
        result = self.books
        if title is not None:
            result = result[result.book_title.str.contains(title)]

        if author is not None:
            result = result[result.book_author.str.contains(author)]

        return result.sort_values("n_ratings", ascending=False)

    def get_books_rated_by_user(self, user_id):
        return self.ratings[self.ratings.user_id == user_id].merge(self.books, on="ISBN")

    def insert_new_user(self, age=40, city="nyc", state="new york", country="usa"):
        new_user_id = self.users.user_id.max() + 1
        user_row = pd.DataFrame([[new_user_id, age, city, state, country]], columns=["user_id", "age", "city", "state", "country"])
        self.users = pd.concat([user_row, self.users])
        return new_user_id

    def insert_rating(self, user_id, ISBN, rating_val):
        rating_row = pd.DataFrame([[user_id, ISBN, rating_val]], columns=["user_id", "ISBN", "book_rating"])
        self.ratings = pd.concat([rating_row, self.ratings])


class User:
    def __init__(self, ds):
        self.ds = ds
        self.uid = ds.insert_new_user()
        self.name = "user"
        self.add_user_books()

    @abstractmethod
    def add_user_books(self):
        ...


    def recommend_books(self, algo, threshold=10, n=30):
        """Only works with surprise algorithms"""

        books = self.ds.books

        # don't recommend books with few ratings
        # problem: cold-start
        books = books[books.n_ratings >= threshold]

        # don't recommend books already rated
        already_rated = self.ds.get_books_rated_by_user(self.uid)
        books = books[~books.ISBN.isin(already_rated.ISBN)]

        books["pred_rating"] = books.ISBN.apply(lambda isbn: algo.predict(self.uid, isbn).est)
        recc = books.sort_values("pred_rating", ascending=False)[:n]

        print(f"Books read by user {self.name}:\n")
        display(already_rated)

        # print(f"{'Book title': <40} {'Book author': <40} {'User rating'}")
        # for _, row in already_rated.iterrows():
        # print(f"{row['book_title']: <40} {row['book_author']: <40} {row['book_rating']}")

        print("\nRecommended books:\n")
        display(recc)

        # print(f"{'Book title': <40} {'Book author': <40} {'Total book ratings': <40} {'Predicted rating'}")
        # for _, row in :
        # print(f"{row['book_title']: <40} {row['book_author']: <40} {row['n_ratings']: <40} {row['pred_rating']}")

    def books_rated(self):
        return self.ds.get_books_rated_by_user(self.uid)


class FantasyUser(User):
    def __init__(self, ds):
        super().__init__(ds)

    def add_user_books(self):
        # harry potter, sorcerer's stone
        self.ds.insert_rating(self.uid, "059035342x", 10)

        # the hobbit : the enchanting prelude to the lord of the rings
        self.ds.insert_rating(self.uid, "0345339681", 10)

        # a game of thrones (a song of ice and fire, book 1)
        self.ds.insert_rating(self.uid, "0553573403", 10)

        # my sister's keeper : a novel (picoult, jodi)	jodi picoult
        self.ds.insert_rating(self.uid, "0743454529", 1)

        # 84 charing cross road	helene hanff	1990
        self.ds.insert_rating(self.uid, "0140143505", 1)

        # # harry potter, sorcerer's stone
        # self.ds.insert_rating(self.uid, "059035342x", 10)

        # # the hobbit : the enchanting prelude to the lord of the rings
        # self.ds.insert_rating(self.uid, "0345339681", 10)

        # # a game of thrones (a song of ice and fire, book 1)
        # self.ds.insert_rating(self.uid, "0553573403", 10)

        # # harry potter, sorcerer's stone
        # self.ds.insert_rating(self.uid, "059035342x", 10)

        # # harry potter and the order of phoenix
        # self.ds.insert_rating(self.uid, "043935806x", 10)

        # # the fellowship of the ring (the lord of the rings, part 1)
        # self.ds.insert_rating(self.uid, "0345339703", 10)

        # # the two towers (the lord of the rings, part 2)
        # self.ds.insert_rating(self.uid, "0345339711", 10)

        # # the lion, the witch, and the wardrobe (the chronicles of narnia, book 2)
        # self.ds.insert_rating(self.uid, "0064471047", 10)


class ScifiUser(User):
    def __init__(self, ds):
        super().__init__(ds)

    def add_user_books(self):
        # Dune, frank herbert
        self.ds.insert_rating(self.uid, "0441172717", 10)

        # Neuromancer, william gibson
        self.ds.insert_rating(self.uid, "0441569595", 10)

        # Ender's game
        self.ds.insert_rating(self.uid, "0812550706", 10)

        # I, robot (Isaac Asimov)
        self.ds.insert_rating(self.uid, "0553294385", 10)

        # Foundations (Isaac Asimov)
        self.ds.insert_rating(self.uid, "0553293354", 10)

        # Stranger in a strange land (Robert A. Heinlein)
        self.ds.insert_rating(self.uid, "0441790348", 10)


class RomanceUser(User):
    def __init__(self, ds):
        super().__init__(ds)

    def add_user_books(self):

        # Pride and prejudice (Jane Austen)
        self.ds.insert_rating(self.uid, "0553213105", 10)

        # Jane Eyre (Charlotte Bronte)
        self.ds.insert_rating(self.uid, "0553211404", 10)


if __name__ == "__main__":
    ds = BookRatingDataset()
