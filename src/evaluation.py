from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score

def evaluate_classification(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro")
    }

def evaluate_regression(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
