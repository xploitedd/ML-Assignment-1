import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import StratifiedKFold

train_data = np.loadtxt('TP1_train.tsv', delimiter='\t')
train_data = shuffle(train_data)

test_data = np.loadtxt('TP1_test.tsv', delimiter='\t')
test_data = shuffle(test_data)

Xs_train = train_data[:,0:-1]
Ys_train = train_data[:,-1]

Xs_test = test_data[:,0:-1]
Ys_test = test_data[:,-1]

# standardize the data
stds = np.std(Xs_train, axis=0)
means = np.std(Xs_train, axis=0)
Xs_train = (Xs_train - means) / stds

stds = np.std(Xs_test, axis=0)
means = np.mean(Xs_test, axis=0)
Xs_test = (Xs_test - means) / stds

class NaiveBayesKde:
    def __init__(self):
        self.best_bw = 0
        self.best_valid_err = 1
        self.trained = False

    def _kde_estimate(self, fit_x, train_x, valid_x, bw):
        kde = KernelDensity(bandwidth=bw).fit(fit_x)
        return kde.score_samples(train_x), kde.score_samples(valid_x)

    def _kde(self, fit_x, test_x):
        kde = KernelDensity(bandwidth=self.best_bw).fit(fit_x)
        return kde.score_samples(test_x)

    def _kde_calc_fold(self, Xs, Ys, tr_ix, va_ix, bw):
        self._x_c0 = Xs[tr_ix[Ys[tr_ix] == 0]]
        self._x_c1 = Xs[tr_ix[Ys[tr_ix] == 1]]

        # probability of the classes
        self._p_c0 = c_train_0 = c_valid_0 = np.log(len(tr_ix[Ys[tr_ix] == 0]) / len(tr_ix))
        self._p_c1 = c_train_1 = c_valid_1 = np.log(len(tr_ix[Ys[tr_ix] == 1]) / len(tr_ix))

        for feat in range(0, Xs.shape[1]):
            # calculate logarithmic density for class 0, i.e log(P(x_i | c_0))
            t_dens, v_dens = self._kde_estimate(self._x_c0[:,[feat]], Xs[tr_ix][:,[feat]], Xs[va_ix][:,[feat]], bw)
            c_train_0 += t_dens
            c_valid_0 += v_dens

            # calculate logarithmic density for class 1, i.e log(P(x_i | c_1))
            t_dens, v_dens = self._kde_estimate(self._x_c1[:,[feat]], Xs[tr_ix][:,[feat]], Xs[va_ix][:,[feat]], bw)
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

    def fit(self, Xs, Ys, folds, start_bw, end_bw, step_bw):
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
        
        self.trained = True
        return self.best_bw, self.best_valid_err

    def predict(self, Xs):
        if self.trained == False:
            raise Exception('The classified is not trained')

        c_test_0 = self._p_c0
        c_test_1 = self._p_c1

        for feat in range(0, Xs.shape[1]):
            test_dens = self._kde(self._x_c0[:,[feat]], Xs[:,[feat]])
            c_test_0 += test_dens

            test_dens = self._kde(self._x_c1[:,[feat]], Xs[:,[feat]])
            c_test_1 += test_dens

        predictions = np.argmax([c_test_0, c_test_1], axis=0)
        return predictions

    def test_error(self, predictions, Ys):
        cmp_predict = Ys == predictions
        test_error = (len(Ys) - np.sum(cmp_predict)) / len(Ys)
        return test_error

nb_kde = NaiveBayesKde()
best_bw, best_valid_err = nb_kde.fit(Xs_train, Ys_train, 5, 0.02, 0.61, 0.02)

print("Best bandwidth is", best_bw, "with validation error", best_valid_err)

predictions = nb_kde.predict(Xs_test)
test_error = nb_kde.test_error(predictions, Ys_test)

print("Naive Bayes w/ KDE - Test Error:", test_error)