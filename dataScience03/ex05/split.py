import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import sys



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
70-80% training gives enough data to learn patterns
20-30% is enough for statistically significant evaluation
INDUSTRY STANDARD: Most common split ratio in machine learning
'''
def split_training_validation(df, to_save_train_filename, to_save_val_filename):
    """Split df into training and validation set"""

    train_df, val_df = train_test_split(df, test_size=0.3, stratify=df["knight"], random_state=42)

    train_df.to_csv(to_save_train_filename, index=False)
    val_df.to_csv(to_save_val_filename, index=False)


def main():
    """main()"""

    try:
 
        if len(sys.argv) != 2:
            print("Error: the arguments are bad")
            return
    
        filepath = str(sys.argv[1])
    
        if not "_" in filepath:
            print("Error: invalid filename")
            return

        filename = filepath.split("_")[-1]
        to_save_train_filename = "Training_" + filename
        to_save_val_filename = "Validation_" + filename

        train_df = load(filepath)
        
        split_training_validation(train_df, to_save_train_filename, to_save_val_filename)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
