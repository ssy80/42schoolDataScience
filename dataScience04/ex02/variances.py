import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA


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


def plot_line(cumulative_variances):
    """Plot line graph for cumulative variance"""

    components = range(1, len(cumulative_variances) + 1) #[1- 30]

    variance_df = pd.DataFrame({
        "cumulative_variances": cumulative_variances,
        "components": components
        })
    #print(variance_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=variance_df, 
        x='components', 
        y='cumulative_variances'
        )

    plt.tight_layout()
    plt.show()


def variances(df):
    """Count variances"""

    features_df = df.drop(["knight"], axis=1)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features_df.columns)

    #features_var = features_scaled_df.var().sort_values(ascending=False)
    #print(features_var)

    pca = PCA()
    pca.fit(features_scaled_df)

    explained_variance_ratio = pca.explained_variance_ratio_
    variances_percentage = explained_variance_ratio * 100
    print("Variances (Percentage):")
    print(variances_percentage)

    cumulative_variances = np.cumsum(variances_percentage)
    print("Cumulative Variances (Percentage):")
    print(cumulative_variances)

    plot_line(cumulative_variances)


def main():
    """main()"""

    try:
 
        train_knight_filepath = "./Train_knight.csv"

        train_knight_df = load(train_knight_filepath)

        variances(train_knight_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
