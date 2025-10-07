import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn import tree


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
    tree.plot_tree(dt,  # Your fitted DecisionTreeClassifier model
                feature_names=feature_names,  # List of your feature names
                class_names=['Sith', 'Jedi'],  # List of your class names in ascending numerical order
                filled=True,  # Fills nodes with colors representing the majority class
                rounded=True,  # Uses rounded corners for node boxes
                fontsize=12)  # Adjust font size for readability

    plt.title("Your Decision Tree")  # Add a title
    plt.show()  # Display the plot


'''
Decision Trees Don't Need Low VIF
'''
def plot_dt(df, test_knight_filepath):
    """Plot decision tree classifier"""

    ##target = train_knight_df[["knight"]]
    #encoder = LabelEncoder()
    #target_encoded = encoder.fit_transform(target["knight"])  # Jedi=1, Sith=0
    #print(target_encoded)
    #df = train_knight_df[features]

    train_df, val_df = train_test_split(df, test_size=0.3, stratify=df["knight"], random_state=42)
    
    # Separate features and target for training set
    X_train = train_df.drop("knight", axis=1)  # All features except target
    y_train = train_df["knight"]               # Target variable
    
    # Separate features and target for validation set  
    X_test = val_df.drop("knight", axis=1)
    y_test = val_df["knight"]


    # Split data
    #X_train, X_test, y_train, y_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["knight"])

    # Train decision tree
    dt = DecisionTreeClassifier(
        #max_depth=4,  # Limit depth for interpretability
        #min_samples_split=10,
        #min_samples_leaf=5,
        max_depth=4,  # Limit depth for interpretability
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    dt.fit(X_train, y_train)

    # Predictions
    y_pred = dt.predict(X_test)
    print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    f1 = f1_score(y_test, y_pred, average='weighted')  # For multi-class
    print(f1)

    #X = df.drop("knight", axis=1)  # Features from entire dataset
    #y = df["knight"]               # Target from entire dataset
    #cv_scores = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
    #cv_scores_f1 = cross_val_score(dt, X, y, cv=5, scoring='f1_weighted')
    #print(cv_scores_f1.mean())

    #f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'binary' if only 2 classes
    #f1 = f1_score(y_test, y_pred, average='binary')  # Use 'binary' if only 2 classes
    #precision, recall, f1_scores, support = precision_recall_fscore_support(y_test, y_pred)
    #print(f1_scores)
    plot_d_tree(dt, df)


def main():
    """main()"""

    try:
 
        train_knight_filepath = "./Train_knight.csv"
        test_knight_filepath = "./Test_knight.csv"

        train_knight_df = load(train_knight_filepath)

        #print(train_knight_df)
        plot_dt(train_knight_df, test_knight_filepath)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
