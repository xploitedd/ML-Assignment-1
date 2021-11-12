import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

class SvmClassifier:
    def __init__(self):
        self._errors = []
        self._svc = None

    def _svc_err(self, X_r, Y_r, tr_ix, va_ix, c, gamma):
        svc = SVC(kernel='rbf', C=c, gamma=gamma)
        svc.fit(X_r[tr_ix], Y_r[tr_ix])
        
        train_err = 1 - svc.score(X_r[tr_ix], Y_r[tr_ix])
        valid_err = 1 - svc.score(X_r[va_ix], Y_r[va_ix])

        return train_err, valid_err

    def optimize_parameters(self, X_r, Y_r, folds, g_range, c_range):
        kfold = StratifiedKFold(n_splits=folds)

        best_c = best_gamma = 0
        best_valid_err = 1

        for gamma in np.arange(g_range[0], g_range[1] + g_range[2], g_range[2]):
            bg_t_err = bg_v_err = 1

            for c in np.arange(c_range[0], c_range[1] + c_range[2], c_range[2]):
                train_err = valid_err = 0

                for tr_ix, va_ix in kfold.split(Y_r, Y_r):
                    tr_err, va_err = self._svc_err(X_r, Y_r, tr_ix, va_ix, c, gamma)
                    train_err += tr_err / folds
                    valid_err += va_err / folds

                if (valid_err < best_valid_err):
                    best_valid_err = valid_err
                    best_c = c
                    best_gamma = gamma

                if (valid_err < bg_v_err):
                    bg_v_err = valid_err
                    bg_t_err = train_err

            self._errors.append([gamma, bg_t_err, bg_v_err])

        return best_c, best_gamma, best_valid_err

    def fit(self, X_r, Y_r, gamma, c):
        self._svc = SVC(kernel='rbf', C=c, gamma=gamma)
        self._svc.fit(X_r, Y_r)

    def predict(self, X_t):
        if (self._svc == None):
            raise Exception('The classifier was not fitted')
        
        return self._svc.predict(X_t)

    def score(self, X_t, Y_t):
        if (self._svc == None):
            raise Exception('The classifier was not fitted')

        return self._svc.score(X_t, Y_t)

    def plot_errors(self, save_fig=True, fig_name='SVM.png'):
        errors = np.array(self._errors)

        plt.figure()
        plt.title('Training vs Cross Validation Errors')
        plt.plot(errors[:,0], errors[:,1], 'b', label='Training Error')
        plt.plot(errors[:,0], errors[:,2], 'r', label='Cross-Validation Error')
        plt.legend()

        if save_fig:
            plt.savefig(fig_name)

        plt.show()
        return
