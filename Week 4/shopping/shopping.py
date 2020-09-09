import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # A dictionary of mapping month to int
    month = {'May': 4, 'Nov': 10, 'Mar': 2, 'Dec': 11, 'Oct': 9, 'Sep': 8, 'Aug': 7, 'Jul': 6, 'June': 5, 'Feb': 1, 'Jan': 0, 'Apr': 3}

    # A simple function for TRUE FALSE
    tf = lambda x: 1 if x == "TRUE" else 0

    # A dictionary of function to help me deal with each row
    # It will be easier to do this way, though less readable
    func = {"Administrative": int, "Administrative_Duration": float, "Informational": int, "Informational_Duration": float,
            "ProductRelated": int, "ProductRelated_Duration": float, "BounceRates": float, "ExitRates": float, "PageValues": float,
            "SpecialDay": float, "Month": lambda x: month[x], "OperatingSystems": int, "Browser": int, "Region": int,
            "TrafficType": int, "VisitorType": lambda x: 1 if x == "Returning_Visitor" else 0, "Weekend": tf, "Revenue": tf}

    keys = list(func.keys())
    evidence = []
    labels = []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for line in reader:
            tmp = []
            for key in keys[:-1]:
                # Map the evidence with their respective functions
                tmp.append(func[key](line[key]))
            evidence.append(tmp)
            labels.append(tf(line[keys[-1]]))

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Just call from sklearn library
    method = KNeighborsClassifier(n_neighbors=1)
    return method.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # Just counting
    total_positive = 0
    total_negative = 0
    negative_predicted = 0
    positive_predicted = 0

    for label, prediction in zip(labels, predictions):
        if label == 1:
            total_positive += 1
            if label == prediction:
                positive_predicted += 1

        else:
            total_negative += 1
            if label == prediction:
                negative_predicted += 1

    return (positive_predicted / total_positive, negative_predicted / total_negative)


if __name__ == "__main__":
    main()
