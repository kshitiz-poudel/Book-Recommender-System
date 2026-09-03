# Book Recommender System

A book recommendation system built with Python and item-based collaborative filtering.

## Dataset

The project uses three files:

- `Books.csv`
- `Ratings.csv`
- `Users.csv`

The dataset is kept outside the repository because it is large.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas scipy scikit-learn



```

## Running the recommender

```bash
python book_recommender_train.py \
  --data-dir "$HOME/Downloads/ML_dataset" \
  --isbn 0316666343 \
  --n 10
```

The program trains a recommendation model and recommends similar books.

## Example

ISBN `0316666343` represents *The Lovely Bones*.

## Files

- `book_recommender_train.py`: training and recommendation script
- `.gitignore`: excludes the dataset, virtual environment, and generated model
- `README.md`: project documentation


