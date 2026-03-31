from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def compute_f1(y_true, y_pred):
    """
    Macro-averaged F1 score for multi-class classification
    """
    return f1_score(y_true, y_pred, average="macro")


def compute_auroc(y_true, y_proba, n_classes):
    """
    Macro-averaged AUROC using One-vs-Rest strategy
    """
    return roc_auc_score(
        y_true,
        y_proba,
        multi_class="ovr",
        average="macro"
    )
