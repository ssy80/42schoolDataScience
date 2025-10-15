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


'''
A scatterplot is one of the most visual ways to spot correlation between two features.
Here’s how you interpret it:
Positive correlation → points go upward from left to right (like a rising line).
Example: as Empowered increases, Stims also increases.
Negative correlation → points go downward from left to right.
Example: as Empowered increases, Stims decreases.
No correlation → points look like a cloud with no clear pattern.
Strength of correlation →
Tight clustering around a line = strong correlation.
Wide scatter = weak correlation.
Empowered  Stims  knight
0        5.0    2.5    Sith  → Green point at (5.0, 2.5)
1        3.0    4.0    Jedi  → Blue point at (3.0, 4.0)
2        6.0    1.0    Sith  → Green point at (6.0, 1.0)
'''
def plot_scatter(test_knight_df, train_knight_df):
    """Plot scatterplot for selected features"""

    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    axes = axes.flatten()

    #print(train_knight_df[train_knight_df["knight"] == 'Jedi'].head(40))

    sns.scatterplot(data=train_knight_df, x="Empowered", y="Stims", hue="knight", ax=axes[0], alpha=0.5, palette={"Sith": "green", "Jedi": "blue"})
    axes[0].legend(loc='upper left')

    sns.scatterplot(data=train_knight_df, x="Push", y="Blocking", hue="knight", ax=axes[1], alpha=0.5, palette={"Sith": "green", "Jedi": "blue"})
    
    sns.scatterplot(data=test_knight_df, x="Empowered", y="Stims", ax=axes[2], alpha=0.5, label="Knight", color="green")
    axes[2].legend(loc='upper left')

    sns.scatterplot(data=test_knight_df, x="Push", y="Blocking", ax=axes[3], alpha=0.5, label="Knight", color="green")
    
    plt.tight_layout()
    plt.show()


def main():
    """main()"""

    try:
 
        test_knight_filepath = "./Test_knight.csv"
        train_knight_filepath = "./Train_knight.csv"

        test_knight_df = load(test_knight_filepath)
        train_knight_df = load(train_knight_filepath)

        plot_scatter(test_knight_df, train_knight_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
