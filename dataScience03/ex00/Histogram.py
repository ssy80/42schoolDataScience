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


def plot_test_hist(df):
    """Plot histogram of all features"""

    fig, axes = plt.subplots(6, 5, figsize=(22, 14))
    axes = axes.flatten()

    for i, feature in enumerate(df.columns):
        sns.histplot(data=df, x=feature, ax=axes[i], bins=42, color="green", label="Knight")
        axes[i].set_title(f"{feature}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].legend(loc='upper right')

    plt.tight_layout()
    plt.show()


'''
jedi_df = df.query("Knight == 'Jedi'")
jedi_df = df.loc[df['Knight'] == 'Jedi', :]
jedi_strong_df = df[(df['Knight'] == 'Jedi') & (df['Strength'] > 100)]
force_users_df = df[df['Knight'].isin(['Jedi', 'Sith'])]
Overlapping histograms are essentially showing you feature 
importance for classification between Jedi and Sith!
High overlap features: Poor predictors for classification
Low overlap features: Good candidates for your model
Complete separation: Perfect predictors (rare in real data)
Distribution shape: Affects which algorithms will work best
'''
def plot_train_hist(features, df):
    """Plot histogram of a features with target Knight"""

    fig, axes = plt.subplots(6, 5, figsize=(22, 14))
    axes = axes.flatten()

    jedi_df = df[df["knight"] == "Jedi"]
    sith_df = df[df["knight"] == "Sith"]

    for i, feature in enumerate(features):
        sns.histplot(data=jedi_df, x=feature, ax=axes[i], bins=42, color="blue", alpha=0.5, label="Jedi")
        sns.histplot(data=sith_df, x=feature, ax=axes[i], bins=42, color="green", alpha=0.5, label="Sith")

        '''jedi_mean = jedi_df[feature].mean()
        sith_mean = sith_df[feature].mean()
        axes[i].axvline(jedi_mean, color='blue', linestyle='--', linewidth=2, alpha=0.8)
        axes[i].axvline(sith_mean, color='green', linestyle='--', linewidth=2, alpha=0.8)
        axes[i].axvspan(min(jedi_mean, sith_mean), max(jedi_mean, sith_mean), 
                       alpha=0.1, color='purple', label='Overlap Zone')'''

        axes[i].set_title(f"{feature}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def main():
    """main()"""

    try:
 
        test_knight_filepath = "./Test_knight.csv"
        train_knight_filepath = "./Train_knight.csv"

        test_knight_df = load(test_knight_filepath)
        plot_test_hist(test_knight_df)

        features = test_knight_df.columns

        train_knight_df = load(train_knight_filepath)
        print(train_knight_df)
        plot_train_hist(features, train_knight_df)
        
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()