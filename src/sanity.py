# src/sanity.py
from src.data_load import load_cre_csv, time_split, CRE_TARGET, TIME_COL

if __name__ == "__main__":
    df = load_cre_csv()
    ds = time_split(df)
    print("Rows:", len(df))
    print("Train:", ds.X_train.shape, "Test:", ds.X_test.shape)
    print("Time range train:", ds.X_train[TIME_COL].min(), "->", ds.X_train[TIME_COL].max())
    print("Time range test :", ds.X_test[TIME_COL].min(), "->", ds.X_test[TIME_COL].max())
    print("Target stats train:", ds.y_train.describe())