import numpy as np

from sklearn.metrics import confusion_matrix


def mean_per_class_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
   """
      Calculates the mean per class accuracy by calculating
      the accuracy for each individual class and then averaging
      them. See the links below for more details:

      - https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
   """
   mat = confusion_matrix(preds, targets)

   # Summing over rows results in the total number of elements for each class.
   # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
   class_sums = mat.sum(axis=0)
   per_class_accuracy = mat.diagonal() / class_sums

   return per_class_accuracy.mean()
