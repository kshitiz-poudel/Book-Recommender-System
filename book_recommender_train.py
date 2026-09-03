#!/usr/bin/env python3
"""Train and query a simple item-based book recommender.

Usage:
  python book_recommender_train.py --data-dir ./ML_dataset --isbn 0316666343
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def load_data(data_dir: Path):
    books = pd.read_csv(data_dir / "Books.csv", encoding="latin-1", dtype={"ISBN": "string"}, low_memory=False)
    ratings = pd.read_csv(data_dir / "Ratings.csv", encoding="latin-1", dtype={"ISBN": "string"})
    ratings = ratings[ratings["Book-Rating"] > 0].copy()
    ratings = ratings.merge(books[["ISBN", "Book-Title", "Book-Author"]], on="ISBN", how="inner")
    return books, ratings


def train(data_dir: Path, min_user_ratings: int = 5, min_book_ratings: int = 5):
    books, ratings = load_data(data_dir)

    # Remove cold-start users/books so similarity is based on meaningful histories.
    for _ in range(2):
        user_counts = ratings["User-ID"].value_counts()
        book_counts = ratings["ISBN"].value_counts()
        ratings = ratings[ratings["User-ID"].isin(user_counts[user_counts >= min_user_ratings].index)]
        ratings = ratings[ratings["ISBN"].isin(book_counts[book_counts >= min_book_ratings].index)]

    user_codes, users = pd.factorize(ratings["User-ID"])
    book_codes, isbns = pd.factorize(ratings["ISBN"])
    matrix = csr_matrix(
        (ratings["Book-Rating"].astype(float), (user_codes, book_codes)),
        shape=(len(users), len(isbns)),
    )

    # Cosine similarity between books based on users' positive ratings.
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=21, n_jobs=-1)
    model.fit(matrix.T)

    book_info = books.drop_duplicates("ISBN").set_index("ISBN")[["Book-Title", "Book-Author"]]
    artifact = {"model": model, "matrix": matrix, "users": users, "isbns": isbns, "book_info": book_info}
    return artifact, ratings


def recommend(artifact, isbn: str, n: int = 10):
    isbns = pd.Series(artifact["isbns"])
    matches = isbns[isbns.astype(str) == str(isbn)]
    if matches.empty:
        raise ValueError("ISBN is not in the filtered training set. Try a popular ISBN or lower the minimum counts.")
    idx = int(matches.index[0])
    distances, indices = artifact["model"].kneighbors(artifact["matrix"].T[idx], n_neighbors=min(n + 1, len(isbns)))
    rows = []
    for distance, neighbor_idx in zip(distances[0], indices[0]):
        neighbor_isbn = str(isbns.iloc[neighbor_idx])
        if neighbor_idx == idx:
            continue
        title = artifact["book_info"].loc[neighbor_isbn, "Book-Title"] if neighbor_isbn in artifact["book_info"].index else ""
        author = artifact["book_info"].loc[neighbor_isbn, "Book-Author"] if neighbor_isbn in artifact["book_info"].index else ""
        rows.append({"ISBN": neighbor_isbn, "Book-Title": title, "Book-Author": author, "similarity": round(1 - float(distance), 4)})
    return pd.DataFrame(rows[:n])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--isbn", type=str, required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--min-user-ratings", type=int, default=5)
    parser.add_argument("--min-book-ratings", type=int, default=5)
    parser.add_argument("--save", type=Path, default=Path("book_recommender.pkl"))
    args = parser.parse_args()

    artifact, filtered = train(args.data_dir, args.min_user_ratings, args.min_book_ratings)
    with args.save.open("wb") as f:
        pickle.dump(artifact, f)

    print(f"Training rows: {len(filtered):,}")
    print(f"Users: {len(artifact['users']):,}; books: {len(artifact['isbns']):,}")
    print(f"Saved model: {args.save}")
    print("\nRecommendations:")
    print(recommend(artifact, args.isbn, args.n).to_string(index=False))


if __name__ == "__main__":
    main()
