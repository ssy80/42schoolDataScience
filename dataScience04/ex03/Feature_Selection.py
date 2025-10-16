import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
Variance Inflation Factor (VIF) is used to detect multicollinearity in regression analysis. 
Multicollinearity occurs when two or more independent variables in a regression model are highly correlated with each other.
    Identify redundant features that provide the same information
    Improve model stability and interpretability
    Prevent overfitting in regression models
Iteratively removes features with high VIF until all features meet the threshold
VIF (Variance Inflation Factor) measures how much a feature is correlated with other features
Tolerance = 1/VIF (lower tolerance = higher multicollinearity)
    VIF < 5: Low multicollinearity (good)
    VIF 5-10: Moderate multicollinearity
    VIF > 10: High multicollinearity (problematic)
Treats feature i as the dependent variable
Regresses it against ALL other features
'''
def cal_vif(df, vif_threshold):
    """Calculate Variance Inflation Factor (VIF) for each feature in df"""
    
    features_df = df.drop("knight", axis=1)
    
    while True:

        vif_df = pd.DataFrame()
        vif_df["feature"] = features_df.columns

        vif_df["vif"] = [variance_inflation_factor(features_df.values, i)
                        for i in range(len(features_df.columns))]

        vif_df = vif_df.sort_values(by="vif", ascending=False) # biggest value on top

        if len(vif_df) > 0 and vif_df.iloc[0]["vif"] >= vif_threshold:
            features_df = features_df.drop(vif_df.iloc[0]["feature"], axis=1) # Delete feature
        else:
            break

    vif_df["tolerance"] = 1 / vif_df["vif"]
    print(vif_df)


def main():
    """main()"""

    try:
 
        train_knight_filepath = "./Train_knight.csv"

        train_knight_df = load(train_knight_filepath)

        cal_vif(train_knight_df, 5)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
