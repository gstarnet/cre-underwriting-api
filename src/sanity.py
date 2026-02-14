# src/sanity.py
from src.data_load import load_cre_csv, time_split, TIME_COL

if __name__ == "__main__":
    df = load_cre_csv()
    ds = time_split(df)

    print("Rows:", len(df))
    print("Train:", ds.X_train.shape, "Test:", ds.X_test.shape)

    # TIME_COL is dropped from X by design, so use df to print ranges
    df_sorted = df.sort_values(TIME_COL).reset_index(drop=True)
    n_train = len(ds.X_train)

    train_range = (df_sorted.iloc[:n_train][TIME_COL].min(), df_sorted.iloc[:n_train][TIME_COL].max())
    test_range = (df_sorted.iloc[n_train:][TIME_COL].min(), df_sorted.iloc[n_train:][TIME_COL].max())

    print("Time range train:", train_range[0], "->", train_range[1])
    print("Time range test :", test_range[0], "->", test_range[1])

    print("Target stats train:", ds.y_train.describe())