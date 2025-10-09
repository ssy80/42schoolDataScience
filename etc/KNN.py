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


def plot_knn(df):
    """Plot line graph for KNN result"""

    fig, ax = plt.subplots(figsize=(8,5))

    sns.lineplot(data=df, x="k", y="score")

    plt.xlabel("k values")
    plt.ylabel("accuracy")
    plt.show()


def simple_knn_search(X_train, y_train, X_val, y_val, k_range=range(1, 31)):
    """Simple search using validation set"""
    
    results = {}
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        
        results[k] = {
            'precision': precision_score(y_val, y_pred, average='macro'),
            'accuracy': accuracy_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred, average='macro')
        }
    
    return results


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

    '''results = simple_knn_search(X_train, y_train, X_test, y_test, k_range=range(1, 31))
    print(results)
    # Convert results to DataFrame for seaborn
    results_df = pd.DataFrame.from_dict(results, orient='index')
    #results_df = pd.DataFrame.from_dict(results)
    print(results_df)
    results_df = results_df.reset_index()
    print(results_df)
    results_df = results_df.rename(columns={'index': 'k'})
    print(results_df)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, x='k', y='f1')
    plt.title('KNN Accuracy vs K Value', fontsize=14, fontweight='bold')
    plt.xlabel('K Value', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(range(1, 31))  # Show all K values on x-axis
    plt.grid(True, alpha=0.3)
    plt.show()'''
    
    # KNN
    knn = KNeighborsClassifier()

    # create param_grid to supply the knn k values
    n_range = range(1, 31) #[1,2,3 ....30]
    param_grid = {
        "n_neighbors": n_range
        }

    # Create separate GridSearchCV for each metric
    scoring_params = {
        "precision": "precision_macro",
        "accuracy": "accuracy",
        "f1": "f1_macro"
    }

    best_params = {}
    best_scores = {}
    results = {}
    grids = {}

    for metric, score in scoring_params.items():

        # Create the GridSearchCV object
        grid_search = GridSearchCV(
            estimator=knn,
            param_grid=param_grid,              
            scoring=score,
            cv=5,                              # Number of cross-validation folds
            n_jobs=-1                          # Use all CPUs
        )

        grid_search.fit(X_train_scaled, y_train)

        best_params[metric] = grid_search.best_params_["n_neighbors"]
        best_scores[metric] = grid_search.best_score_

        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df = results_df[["param_n_neighbors", "mean_test_score"]]
        results_df.columns = ["k", "score"]
        results[metric] = results_df
        
        print(f"{metric} k: {best_params[metric]}")
        print(f"{metric} score: {(best_scores[metric]*100):.2f}%")
        print(results[metric])

        grids[metric] = grid_search

    # Predict using the Test_knight.csv
    best_knn = grids["precision"].best_estimator_
    X_test_scaled = scaler.transform(test_df)
    y_pred = best_knn.predict(X_test_scaled)
    pd.DataFrame(y_pred).to_csv("KNN.txt", index=False, header=None)

    plot_knn(results["accuracy"])


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
