from sklearn.metrics import confusion_matrix, recall_score

def model_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true= y_true, y_pred= y_pred).ravel()

    recall = recall_score(y_true= y_true, y_pred= y_pred)

    tpr = tp / (tp + fn)  #True Positive Rate 
    fpr = fp / (fp + tn)  #False Positive Rate


    return{
        'tp' : tp,
        'tn' : tn,
        'fp' : fp,
        'fn' : fn,
        'tpr' : tpr,
        'fpr' : fpr,
        'recall' : recall
    }