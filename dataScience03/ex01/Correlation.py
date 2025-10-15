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
Correlation indicates the strength and direction of a linear 
relationship between a feature and the target variable
Correlation helps in feature selection by identifying features
with a strong relationship to the target for prediction, or
by detecting redundant features with high correlation to each other. 
Remove redundant features: If two variables are highly correlated,
they are providing similar information. To reduce redundancy and
simplify the model, you can remove one of these features. 
'''
def corr_knight(df):
    """Encode knight(object) to numeric, print correlations between features and target"""

    df['knight_encoded'] = df['knight'].map({'Sith': 0, 'Jedi': 1})

    df = df.drop("knight", axis=1)
    df.rename(columns={"knight_encoded": "knight"}, inplace=True)

    correlations = df.corr()["knight"].sort_values(ascending=False)
    print(correlations)


def main():
    """main()"""

    try:
 
        train_knight_filepath = "./Train_knight.csv"

        train_knight_df = load(train_knight_filepath)

        corr_knight(train_knight_df)
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
