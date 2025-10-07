import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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


def plot_heatmap(df):
    """Plot heatmap"""

    knight = df['knight']
    df['knight'] = knight.map({'Sith': 0, 'Jedi': 1})

    correlation_matrix = df.corr()

    fig, ax = plt.subplots(figsize=(16, 12))

    sns.heatmap(correlation_matrix, 
            fmt='.2f',
            cmap="Blues",
            square=True,
            cbar=True
            )
    
    plt.tight_layout()
    plt.show()


def main():
    """main()"""

    try:
 
        train_knight_filepath = "./Train_knight.csv"

        train_knight_df = load(train_knight_filepath)

        plot_heatmap(train_knight_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
