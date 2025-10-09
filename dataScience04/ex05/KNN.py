import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import precision_score, accuracy_score, f1_score


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


def plot_knn(df, metric):
    """Plot line graph for KNN result"""

    fig, ax = plt.subplots(figsize=(8,5))

    sns.lineplot(data=df, x="k", y=metric)

    plt.xlabel("k values")
    plt.ylabel(metric)
    plt.show()


'''
K-Nearest Neighbors (KNN) is a simple yet powerful machine learning algorithm 
used for both classification and regression tasks. It's called an instance-based 
or lazy learning algorithm because it doesn't build an explicit model during training
- it simply stores all the training data.
GridSearchCV can be used for all classifier/regression
scoring='precision_macro' is a way to calculate precision for multi-class classification problems 
'''
def knn(training_df, val_df, test_df):
    """Perform KNN classifying"""

    # Separate features and target for training set
    X_train = training_df.drop("knight", axis=1)
    y_train = training_df["knight"] # "Sith" or "Jedi"
    
    # Separate features and target for validation set
    X_test = val_df.drop("knight", axis=1)
    y_test = val_df["knight"]

    # scale values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    k_range=range(1, 31)

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        
        results[k] = {
            "precision": precision_score(y_test, y_pred, average="macro"),
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="macro")
        }

    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df = results_df.reset_index()
    results_df = results_df.rename(columns={"index": "k"})
    print(results_df)

    plot_knn(results_df, "precision")

    # find knn with best precision
    best_row_id = results_df["precision"].idxmax()
    best_row = results_df.iloc[best_row_id]
    best_k = int(best_row["k"])
    print(f"Best k: {best_k} (base on precision), precision: {best_row["precision"]*100:.2f}%, f1: {best_row["f1"]*100:.2f}%, accuracy: {best_row["accuracy"]*100:.2f}%")

    # Predict the Test_knight.csv
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train_scaled, y_train)
    X_test_scaled = scaler.transform(test_df)
    y_pred = best_knn.predict(X_test_scaled)
    pd.DataFrame(y_pred).to_csv("KNN.txt", index=False, header=None)


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
        
        knn(training_df, val_df, test_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
