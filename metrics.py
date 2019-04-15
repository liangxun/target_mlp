# reference url: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt


class Metrics:
    def __init__(self, y_true, prediction):
        self.y_true = y_true
        self.y_pred = [np.argmax(item) for item in prediction]
        self.y_score1 = prediction[:, 1]
        self.y_score0 = prediction[:, 0]

    def confusion_matrix(self):
        return confusion_matrix(y_true=self.y_true, y_pred=self.y_pred)

    def precision(self):
        return precision_score(y_true=self.y_true, y_pred=self.y_pred)

    def recall(self):
        return recall_score(y_true=self.y_true, y_pred=self.y_pred)

    def F_score(self):
        return f1_score(y_true=self.y_true, y_pred=self.y_pred)

    def roc_curve(self, show=True):
        fpr, tpr, thresholds = roc_curve(y_true=self.y_true, y_score=self.y_score1)
        if show is True:
            plt.plot(fpr, tpr)
            plt.show()
        return fpr, tpr, thresholds

    def auc(self):
        fpr, tpr, threshold = self.roc_curve(show=False)
        return auc(fpr, tpr)

    def describe(self, show=False):
        print("precision=", self.precision())
        print("recall=", self.recall())
        print("F1_score=", self.F_score())
        print("auc=", self.auc())
        print("confusion_matrix:\n", self.confusion_matrix())
        if show is True:
            self.roc_curve()


if __name__ == '__main__':
    label = np.load("./out/test_label.npy")
    prediction = np.load("./out/prediction.npy")

    print(label.shape)
    print(prediction.shape)
    M = Metrics(label, prediction)
    M.describe()
