import numpy as np

def normal_test(Ys_true, Ys_pred):
    size = len(Ys_true)
    cmp = Ys_true == Ys_pred
    errors = size - np.sum(cmp)

    sigma = np.sqrt(errors * (1 - errors / size))
    return errors, 1.96 * sigma

def mcnemar_test(Ys_true, Ys_pred_1, Ys_pred_2):
    cmp = [0, 0]
    for i in range(0, len(Ys_true)):
        if (Ys_pred_1[i] == Ys_true[i] and Ys_pred_2[i] != Ys_true[i]):
            cmp[0] += 1
        elif (Ys_pred_1[i] != Ys_true[i] and Ys_pred_2[i] == Ys_true[i]):
            cmp[1] += 1

    diff = np.abs(cmp[0] - cmp[1])
    csum = cmp[0] + cmp[1]

    return ((diff - 1) ** 2) / csum