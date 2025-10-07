import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys


def load(path: str, names: list) -> pd.DataFrame:
    """
    Load a csv file path into a dataframe,
    return the dataframe if success, else return None.
    """
    try:
        df = pd.read_csv(path, header=None, names=names)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None
    return df


def plot_heatmap(confusion_array):
    """Plot heatmap of confusion matrix"""

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(confusion_array, 
            annot=True,           # Show numbers in cells
            fmt='d',              # Format as integers
            cbar=True)            # Show color bar

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


'''
A confusion matrix is a table used to evaluate the performance of a classification model
— it shows how well your model’s predictions match the true labels.
Accuracy	            (TP + TN) / (TP + TN + FP + FN)	Overall correctness
Precision	            TP / (TP + FP)	When it predicts “Yes”, how often is it correct
Recall (Sensitivity)	TP / (TP + FN)	How many actual “Yes” it correctly finds
F1-score	            2 × (Precision × Recall) / (Precision + Recall)	Balance between precision and recall
Each class has its own set of TP, FP, FN, TN depending on which class you treat as “positive”.
truth/predict    Jedi   Sith
Jedi           TP        FN  
Sith           FP        TN
'''
def confusion_matrix(predictions_df, truth_df):
    """Calculate confusion matrix"""

    combined_df = pd.concat([truth_df, predictions_df], axis=1)

    #positive = "Jedi"
    labels = ["Jedi", "Sith"]
    confusion_df = pd.DataFrame(0, index=labels, columns=labels)

    for i, row in combined_df.iterrows():
        truth = row["truth"]
        predict = row["predictions"]
        confusion_df.loc[truth, predict] += 1

    print(confusion_df)
    confusion_array = confusion_df.to_numpy()
    print(confusion_array)
    
    tp_jedi = confusion_df.loc["Jedi", "Jedi"]  # True Positive: Actual Jedi → Predicted Jedi
    fn_jedi = confusion_df.loc["Jedi", "Sith"]  # False Negative: Actual Jedi → Predicted Sith
    fp_jedi = confusion_df.loc["Sith", "Jedi"]  # False Positive: Actual Sith → Predicted Jedi
    tn_jedi = confusion_df.loc["Sith", "Sith"]  # True Negative: Actual Sith → Predicted Sith

    jedi_precision = round(tp_jedi / (tp_jedi + fp_jedi), 2)
    jedi_recall =  round(tp_jedi / (tp_jedi + fn_jedi), 2)
    jedi_f1 = round((2 * (jedi_precision * jedi_recall)) / (jedi_precision + jedi_recall), 2)
    jedi_total = confusion_df.loc["Jedi", "Jedi"] + confusion_df.loc["Jedi", "Sith"]
    
    tp_sith = confusion_df.loc["Sith", "Sith"]  # True Positive: Actual Sith → Predicted Sith
    fn_sith = confusion_df.loc["Sith", "Jedi"]  # False Negative: Actual Sith → Predicted Jedi
    fp_sith = confusion_df.loc["Jedi", "Sith"]  # False Positive: Actual Jedi → Predicted Sith
    tn_sith = confusion_df.loc["Jedi", "Jedi"]  # True Negative: Actual Jedi → Predicted Jedi

    sith_precision = round(tp_sith / (tp_sith + fp_sith), 2)
    sith_recall =  round(tp_sith / (tp_sith + fn_sith), 2)
    sith_f1 = round((2 * (sith_precision * sith_recall)) / (sith_precision + sith_recall), 2)
    sith_total = confusion_df.loc["Sith", "Jedi"] + confusion_df.loc["Sith", "Sith"]

    total = tp_sith + fn_sith + fp_sith + tn_sith
    accuracy = (tp_sith + tp_jedi) / total

    summary_index = ["Jedi", "Sith", "accuracy"]
    precision_col = [jedi_precision, sith_precision, ""]
    recall_col = [jedi_recall, sith_recall, ""]
    f1_col = [jedi_f1, sith_f1, ""]
    total_col = [jedi_total, sith_total, ""]

    summary_df = pd.DataFrame({
        "precision": precision_col,
        "recall": recall_col,
        "f1-score": f1_col,
        "total": total_col
        }, index=summary_index)

    summary_df.loc["accuracy", "f1-score"] = accuracy
    summary_df.loc["accuracy", "total"] = total
    
    print(summary_df)

    plot_heatmap(confusion_array)


def main():
    """main()"""

    try:
 
        if len(sys.argv) != 3:
            print("Error: the arguments are bad")
            return
    
        predictions_filepath = str(sys.argv[1])
        truth_filepath = str(sys.argv[2])

        predictions_df = load(predictions_filepath, names=["predictions"])
        truth_df = load(truth_filepath, names=["truth"])

        confusion_matrix(predictions_df, truth_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
