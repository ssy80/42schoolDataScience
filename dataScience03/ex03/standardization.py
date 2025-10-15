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


def plot_scatter(df):
    """Plot scatterplot for selected features"""

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(data=df, x="Empowered", y="Stims", hue="knight", alpha=0.5, palette={"Sith": "green", "Jedi": "blue"})
    
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def standardization_plot(train_knight_df):
    """Plot scatter plot after standardization scale"""

    knight_df = train_knight_df["knight"]
    
    features_df = train_knight_df.copy()
    features_df = features_df.drop("knight", axis=1)
    features_df_columns = features_df.columns

    # scale
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features_df))
    features_scaled.columns = features_df_columns

    # subject example
    print(features_df.iloc[[360],:])
    print("-" * 50)
    print(features_scaled.iloc[[360],:])

    # add back knight col
    features_scaled = pd.concat([features_scaled, knight_df], axis=1)
    plot_scatter(features_scaled)

    #features_df = pd.concat([features_df, knight_df], axis=1)
    #plot_scatter(features_df)


def main():
    """main()"""

    try:
 
        train_knight_filepath = "./Train_knight.csv"

        train_knight_df = load(train_knight_filepath)

        standardization_plot(train_knight_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
