import pandas as pd


def split_n_pickle(organism: str) -> None:
    df: pd.DataFrame = pd.read_pickle(f"./data/{organism}/cleanedData.pkl")
    seed = 42
    train_df = df.sample(frac=0.8, random_state=seed)
    test_valid_df = df.drop(train_df.index)
    test_df = test_valid_df.sample(frac=0.5, random_state=seed)
    valid_df = test_valid_df.drop(test_df.index)

    print(f"{organism:<25} total: {df.shape[0]:6} | train: {train_df.shape[0]:6} | test: {test_df.shape[0]:6} | valid: {valid_df.shape[0]:6}")

    train_df.to_pickle(f"./data/{organism}/cleanedData_train.pkl")
    test_df.to_pickle(f"./data/{organism}/cleanedData_test.pkl")
    valid_df.to_pickle(f"./data/{organism}/cleanedData_valid.pkl")


organisms = ["E.Coli", "Drosophila.Melanogaster", "Homo.Sapiens"]
for organism in organisms:
    split_n_pickle(organism)
