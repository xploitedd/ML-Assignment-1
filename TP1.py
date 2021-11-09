import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import StratifiedKFold

data = np.loadtxt('TP1_train.tsv', delimiter='\t')
data = shuffle(data)

Xs = data[:,0:-1]
Ys = data[:,-1]

# standardize the data
stds = np.std(Xs, axis=0)
means = np.std(Xs, axis=0)
Xs = (Xs - means) / stds

class NaiveBayesKde:
    def __init__(self):
        self.best_bw = 0
        self.best_valid_err = 1

    def _kde_estimate(self, fit_x, train_x, valid_x, bw):
        kde = KernelDensity(bandwidth=bw).fit(fit_x)
        return kde.score_samples(train_x), kde.score_samples(valid_x)

    def _kde_calc_fold(self, Xs, Ys, tr_ix, va_ix, bw):
        x_c0 = Xs[tr_ix[Ys[tr_ix] == 0]]
        x_c1 = Xs[tr_ix[Ys[tr_ix] == 1]]

        # probability of the classes
        c_train_0 = c_valid_0 = np.log(len(tr_ix[Ys[tr_ix] == 0]) / len(tr_ix))
        c_train_1 = c_valid_1 = np.log(len(tr_ix[Ys[tr_ix] == 1]) / len(tr_ix))

        for feat in range(0, Xs.shape[1]):
            # calculate logarithmic density for class 0, i.e log(P(x_i | c_0))
            t_dens, v_dens = self._kde_estimate(x_c0[:,[feat]], Xs[tr_ix][:,[feat]], Xs[va_ix][:,[feat]], bw)
            c_train_0 += t_dens
            c_valid_0 += v_dens

            # calculate logarithmic density for class 1, i.e log(P(x_i | c_1))
            t_dens, v_dens = self._kde_estimate(x_c1[:,[feat]], Xs[tr_ix][:,[feat]], Xs[va_ix][:,[feat]], bw)
            c_train_1 += t_dens
            c_valid_1 += v_dens

        # calculate predictions
        predict_train = np.argmax([c_train_0, c_train_1], axis=0)
        predict_valid = np.argmax([c_valid_0, c_valid_1], axis=0)

        # compare the predictions with our data sets
        cmp_train = Ys[tr_ix] == predict_train
        cmp_valid = Ys[va_ix] == predict_valid

        # obtain training error and validation error
        train_err = (len(tr_ix) - np.sum(cmp_train)) / len(tr_ix)
        valid_err = (len(va_ix) - np.sum(cmp_valid)) / len(va_ix) 

        return train_err, valid_err

    def train(self, Xs, Ys, folds, start_bw, end_bw, step_bw):
        kfold = StratifiedKFold(n_splits=folds)
        for bw in np.arange(start_bw, end_bw, step_bw):
            train_err = valid_err = 0
            for tr_ix, va_ix in kfold.split(Ys, Ys):
                tr_err, va_err = self._kde_calc_fold(Xs, Ys, tr_ix, va_ix, bw)
                train_err += tr_err / folds
                valid_err += va_err / folds

            if (valid_err < self.best_valid_err):
                self.best_bw = bw
                self.best_valid_err = valid_err
        
        return self.best_bw, self.best_valid_err

nb_kde = NaiveBayesKde()
best_bw, best_valid_err = nb_kde.train(Xs, Ys, 5, 0.02, 0.61, 0.02)

print("Best bandwidth is", best_bw, "with validation error", best_valid_err)