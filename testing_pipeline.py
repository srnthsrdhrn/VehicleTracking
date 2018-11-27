from sklearn.metrics import confusion_matrix
import csv
import numpy as np

array1 = []
array2 = []
with open('pred.csv', 'r') as csvfile:  # predd.csv - CSV File with Predicted values from algorithm
    reader = csv.reader(csvfile)
    for row in reader:
        array1.append(row)
        # print(array1)
timestamp_pred_arr = array1[0]  # array to store timestamp value of predicted values
predicted_arr = array1[1]  # array to store predicted values
with open('actual.csv', 'r') as csvfile:  # actualll.csv - CSV File with actual values manually calculated
    reader = csv.reader(csvfile)
    for row in reader:
        array2.append(row)
        # print(array2)
timestamp_act_arr = array2[0]  # array to store timestamp value of actual(manual) values
actual_arr = array2[1]  # array to store actual(manual) values
for index in range(1, 15):  # 15 is no of predictions made
    if (predicted_arr[index] != actual_arr[index]):
        print("At timestamp %s :" % (timestamp_pred_arr[index]))
        print("%s is wrongly predicted as %s " % (actual_arr[index], predicted_arr[index]))
cm = confusion_matrix(actual_arr, predicted_arr)


def precision(label, confusion_matrix):  # no of correct positive predictions/ total no of positive predictions
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()


def recall(label, confusion_matrix):  # no of pos predictions/total no of positives
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()


def precision_macro_average(confusion_matrix):  # sum of precision/ no of precisions
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows


def recall_macro_average(confusion_matrix):  # sum of recall / no of recall
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns


def accuracy(confusion_matrix):  # accuracy: total number of all correct predictions/ total number of dataset
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements


print("precision total:", precision_macro_average(cm))
print("recall total:", recall_macro_average(cm))
print("Accuracy is:", accuracy(cm))
print("Error rate is:", 1 - accuracy(cm))
