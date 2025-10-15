import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


def load(path: str) -> pd.DataFrame:
    """
    Load a csv file path into a dataframe,
    return the dataframe if success, else return None.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None
    return df


def plot_scatter_train(df):
    """Plot scatterplot for selected features"""

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(data=df, x="Push", y="Blocking", hue="knight", alpha=0.5, palette={"Sith": "green", "Jedi": "blue"})
    
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_scatter_test(df):
    """Plot scatterplot for selected features"""

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(data=df, x="Push", y="Blocking", alpha=0.5, label="Knight", color="green")
    
    plt.tight_layout()
    plt.show()


def normalization_plot(test_knight_df, train_knight_df):
    """Plot scatter plot after normalization scaling"""

    knight_df = train_knight_df["knight"]
    
    train_features_df = train_knight_df.copy()
    train_features_df = train_features_df.drop("knight", axis=1)
    train_features_df_columns = train_features_df.columns

    # scale
    scaler = MinMaxScaler()
    train_features_scaled = pd.DataFrame(scaler.fit_transform(train_features_df))
    train_features_scaled.columns = train_features_df_columns
    
    # subject example
    print(train_features_df.iloc[[360],:])
    print("-" * 50)
    print(train_features_scaled.iloc[[360],:])

    # add back knight col
    train_features_scaled = pd.concat([train_features_scaled, knight_df], axis=1)

    # plot for train_knight_df
    plot_scatter_train(train_features_scaled)

    # scale test_knight_df
    test_features_columns = test_knight_df.columns
    test_features_scaled = pd.DataFrame(scaler.fit_transform(test_knight_df))
    test_features_scaled.columns = test_features_columns
    plot_scatter_test(test_features_scaled)


def main():
    """main()"""

    try:
 
        test_knight_filepath = "./Test_knight.csv"
        train_knight_filepath = "./Train_knight.csv"

        test_knight_df = load(test_knight_filepath)
        train_knight_df = load(train_knight_filepath)

        normalization_plot(test_knight_df, train_knight_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
