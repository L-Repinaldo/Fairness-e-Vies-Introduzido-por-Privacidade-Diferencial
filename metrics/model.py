from sklearn.metrics import confusion_matrix

def compute_model_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true= y_true, y_pred= y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0 #True Positive Rate 
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 #False Positive Rate

    return{
        'tp' : tp,
        'tn' : tn,
        'fp' : fp,
        'fn' : fn,
        'tpr' : tpr,
        'fpr' : fpr,
    }