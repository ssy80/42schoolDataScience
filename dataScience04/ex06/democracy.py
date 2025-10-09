import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, f1_score
from sklearn.pipeline import Pipeline


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
1.0 - 1.5	Well balanced	No special handling needed
1.6 - 2.0	Slightly imbalanced	Usually fine with default settings
2.0 - 5.0	Moderately imbalanced	Consider class_weight='balanced'
5.0+	Severely imbalanced	Definitely use balancing techniques
'''
def imbalance_check(train_df):
    """Check df for imbalance ratio"""

    # Check class distribution
    class_counts = train_df["knight"].value_counts()
    #print("Class distribution:")
    #print(class_counts)

    # Calculate imbalance ratio
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"Imbalance ratio: {imbalance_ratio:.1f}")


def voting(training_df, val_df, test_df):
    """Voting classifier with KNN, Logistic Regression and Decision Tree"""

    # Separate features and target for training set
    X_train = training_df.drop("knight", axis=1)
    y_train = training_df["knight"] # "Sith" or "Jedi"
    
    # Separate features and target for validation set
    X_test = val_df.drop("knight", axis=1)
    y_test = val_df["knight"]

    # Define individual classifiers pipline
    dt_pipeline = Pipeline([
        ("dt", DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ))
    ])

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),   # Scale for Logistic Regression
        ("lr", LogisticRegression(
            penalty="l2",               # Ridge regularization (default)
            C=1.0,                      # Strong regularization (prevents overfitting), Weak regularization (fits training data closely) 
            solver="lbfgs",             # Optimization algorithm
            max_iter=1000,              # Increase if you get convergence warnings
            class_weight=None,          # Handles imbalanced classes
            random_state=42
        ))
    ])

    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=17))
    ])

    # Create Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ("dt", dt_pipeline),
            ("lr", lr_pipeline), 
            ("knn", knn_pipeline)
        ],
        voting="hard"                   # "hard" for majority vote, "soft" for probability average
    )

    voting_clf.fit(X_train, y_train)
    
    # Make predictions with test set
    y_pred = voting_clf.predict(X_test)
    
    # Evaluate performance
    precision = precision_score(y_test, y_pred, average="macro")
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"precision: {precision}")
    print(f"accuracy: {accuracy}")
    print(f"f1: {f1}")
    
    # predict with Test_knight.csv df
    y_pred = voting_clf.predict(test_df)
    pd.DataFrame(y_pred).to_csv("Voting.txt", index=False, header=None)
    

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
        
        imbalance_check(train_df)
        voting(training_df, val_df, test_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
