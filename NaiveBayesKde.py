import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import StratifiedKFold

class NaiveBayesKde:
    def __init__(self):
        self.best_bw = 0
        self.best_valid_err = 1
        self._trained = False
        self._errors = []

    def _kde_estimate(self, fit_x, train_x, valid_x, bw):
        kde = KernelDensity(bandwidth=bw).fit(fit_x)
        return kde.score_samples(train_x), kde.score_samples(valid_x)

    def _kde(self, fit_x, test_x):
        kde = KernelDensity(bandwidth=self.best_bw).fit(fit_x)
        return kde.score_samples(test_x)

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

    def optimize_bandwidth(self, Xs, Ys, folds, start_bw, end_bw, step_bw):
        kfold = StratifiedKFold(n_splits=folds)
        best_bw = 0
        best_valid_err = 1

        for bw in np.arange(start_bw, end_bw, step_bw):
            train_err = valid_err = 0
            for tr_ix, va_ix in kfold.split(Ys, Ys):
                tr_err, va_err = self._kde_calc_fold(Xs, Ys, tr_ix, va_ix, bw)
                train_err += tr_err / folds
                valid_err += va_err / folds

            self._errors.append([bw, train_err, valid_err])
            if (valid_err < best_valid_err):
                best_bw = bw
                best_valid_err = valid_err

        self.best_bw = best_bw
        self.best_valid_err = best_valid_err

        return self.best_bw, self.best_valid_err

    def fit(self, Xs, Ys):
        if self.best_bw == 0:
            raise Exception('The bandwidth should be optimized first')

        self._p_c1 = np.sum(Ys) / len(Ys)
        self._p_c0 = 1 - self._p_c1

        self._x_c0 = Xs[Ys == 0]
        self._x_c1 = Xs[Ys == 1]
        self._trained = True
        return
        
    def predict(self, Xs):
        if self._trained == False:
            raise Exception('The classifier is not trained')

        c_test_0 = self._p_c0
        c_test_1 = self._p_c1

        for feat in range(0, Xs.shape[1]):
            test_dens = self._kde(self._x_c0[:,[feat]], Xs[:,[feat]])
            c_test_0 += test_dens

            test_dens = self._kde(self._x_c1[:,[feat]], Xs[:,[feat]])
            c_test_1 += test_dens

        predictions = np.argmax([c_test_0, c_test_1], axis=0)
        return predictions

    def plot_errors(self, save_fig=True, fig_name='NB.png'):
        errors = np.array(self._errors)

        plt.figure()
        plt.title('Training vs Cross Validation Errors')
        plt.plot(errors[:,0], errors[:,1], 'b', label='Training Error')
        plt.plot(errors[:,0], errors[:,2], 'r', label='Cross-Validation Error')
        plt.legend()
        plt.show()

        if save_fig:
            plt.savefig(fig_name)

        return
