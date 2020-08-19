from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_grid_search(clf, X_train, y_train, C):
    train_auc = clf.cv_results_['mean_train_score']
    train_auc_std = clf.cv_results_['std_train_score']
    cv_auc = clf.cv_results_['mean_test_score']
    cv_auc_std = clf.cv_results_['std_test_score']

    plt.plot(C, train_auc, label='Train Recall Score')
    # this code is copied from here: https://stackoverflow.com/a/48803361/4084039
    plt.gca().fill_between(C, train_auc - train_auc_std, train_auc + train_auc_std, alpha=0.2, color='darkblue')

    plt.plot(C, cv_auc, label='Test Recall Score')
    # this code is copied from here: https://stackoverflow.com/a/48803361/4084039
    plt.gca().fill_between(C, cv_auc - cv_auc_std, cv_auc + cv_auc_std, alpha=0.2, color='darkorange')
    plt.legend()
    plt.xlabel("log(C): hyperparameter")
    plt.ylabel("Recall Score")
    plt.title("ERROR PLOTS")
    plt.grid()
    plt.show()


# This function plots the confusion matrices given y_i, y_i_hat.
def plot_confusion_matrix(train_y, train_predict_y, test_y, test_predict_y):
    # Confusion Matrix
    confusion_matrix_train = confusion_matrix(train_y, train_predict_y)

    # Ploting heatmap of confusion matrix
    # https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
    class_names = ['negative', 'positive']
    confusion_matrix_train = pd.DataFrame(confusion_matrix_train, index=class_names, columns=class_names)
    heatmap = sns.heatmap(confusion_matrix_train, annot=True, fmt='g')

    plt.xlabel('Predicted Class', size=14)
    plt.ylabel('Actual Class', size=14)
    plt.title("Train Confusion Matrix\n", size=24)
    plt.show()

    C = confusion_matrix(test_y, test_predict_y)
    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j

    tn, fp, fn, tp = confusion_matrix(train_y, train_predict_y).ravel()
    print("Train Cost: ", (fp * 10 + fn * 500))
    tn, fp, fn, tp = confusion_matrix(test_y, test_predict_y).ravel()
    print("Test Cost: ", (fp * 10 + fn * 500))

    A = (((C.T) / (C.sum(axis=1))).T)
    # divid each element of the confusion matrix with the sum of elements in that column

    # C = [[1, 2],
    #     [3, 4]]
    # C.T = [[1, 3],
    #        [2, 4]]
    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =1) = [[3, 7]]
    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]
    #                           [2/3, 4/7]]

    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]
    #                           [3/7, 4/7]]
    # sum of row elements = 1

    B = (C / C.sum(axis=0))
    # divid each element of the confusion matrix with the sum of elements in that row
    # C = [[1, 2],
    #     [3, 4]]
    # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =0) = [[4, 6]]
    # (C/C.sum(axis=0)) = [[1/4, 2/6],
    #                      [3/4, 4/6]]
    plt.figure(figsize=(20, 5))
    plt.suptitle("Test Confusion, Presicion & Recall Matrix", fontsize=24)

    labels = ['negative', 'positive']
    # representing A in heatmap format
    plt.subplot(1, 3, 1)
    sns.heatmap(C, annot=True, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")

    plt.subplot(1, 3, 2)
    sns.heatmap(B, annot=True, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Precision matrix")

    plt.subplot(1, 3, 3)
    # representing B in heatmap format
    sns.heatmap(A, annot=True, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Recall matrix")

    plt.show()


def plot_precision_recall_costs(model, data, y_true):
    #     X_train, X_test, y_train, y_test = train_test_split(data, y_true, stratify=y_true, \
    #                                                         test_size=0.266666666666666667, random_state=42)
    X_test = data
    y_test = y_true
    y_pred = model.predict_proba(data)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    thresholds = np.append(thresholds, 1)

    costs = []

    for threshold in tqdm(thresholds):
        y_hat = y_pred > threshold
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        cost = fp * 10 + fn * 500
        costs.append(cost)
    plt.figure(figsize=(20, 6))

    plt.subplot(121)
    plt.plot(thresholds, precision, label='Precision')
    plt.plot(thresholds, recall, label='Recall')
    plt.legend()
    plt.xlabel("Thresholds")
    plt.ylabel("Precision_Recall")
    plt.title("Precision_Recall-Threshold Plot")
    plt.grid()

    plt.subplot(122)
    plt.plot(thresholds, costs, label='Costs')
    plt.legend()
    plt.xlabel("Thresholds")
    plt.ylabel("Costs")
    plt.title("Cost-Threshold Plot")
    plt.grid()

    plt.show()
    print("Best threshold: {:.4f}".format(thresholds[np.argsort(costs)[0]]))
    print("Min cost: {:.2f}".format(costs[np.argsort(costs)[0]]))
    return thresholds[np.argsort(costs)[0]], costs[np.argsort(costs)[0]]