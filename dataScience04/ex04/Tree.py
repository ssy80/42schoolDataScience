import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
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


def plot_d_tree(dt, df):
    """Plot decision tree"""

    feature_names = df.drop("knight", axis=1).columns.tolist()

    fig, ax = plt.subplots(figsize=(18, 10))

    # Plot the decision tree
    tree.plot_tree(
                dt,
                feature_names=feature_names,
                class_names=['Sith', 'Jedi'],
                filled=True,
                rounded=True,
                fontsize=12
                )

    plt.title("Decision Tree trained on all Knights features")
    plt.show()


'''
Decision Trees Don't Need Low VIF
'''
def dt(training_df, val_df, test_df):
    """Perform decision tree classifying"""

    # Separate features and target for training set  
    X_train = training_df.drop("knight", axis=1)
    y_train = training_df["knight"]
    
    # Separate features and target for validation set  
    X_test = val_df.drop("knight", axis=1)
    y_test = val_df["knight"]

    # Train decision tree
    dt = DecisionTreeClassifier(
        criterion='gini',
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    #print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1-score: {f1}")

    # Predict using the Test_knight.csv
    X_test = test_df
    y_pred = dt.predict(X_test)
    pd.DataFrame(y_pred).to_csv("Tree.txt", index=False, header=None)

    plot_d_tree(dt, training_df)


def main():
    """main()"""

    try:

        if len(sys.argv) != 3:
            print("Error: the arguments are bad")
            return
    
        train_filepath = str(sys.argv[1])
        test_filepath = str(sys.argv[2])

        train_df = load(train_filepath)
        test_df = load(test_filepath)

        training_df = load("./Training_knight.csv")
        val_df = load("./Validation_knight.csv")
        
        dt(training_df, val_df, test_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
