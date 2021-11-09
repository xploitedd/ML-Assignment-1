import numpy as np
from NaiveBayesKde import NaiveBayesKde
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB

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

def test_error(predictions, Ys):
    cmp_predict = Ys == predictions
    test_error = (len(Ys) - np.sum(cmp_predict)) / len(Ys)
    return test_error

nb_kde = NaiveBayesKde()
best_bw, best_valid_err = nb_kde.fit(Xs_train, Ys_train, 5, 0.02, 0.61, 0.02)

print("Best bandwidth is", best_bw, "with validation error", best_valid_err)

nb_kde_pred = nb_kde.predict(Xs_test)
nb_kde_test_err = test_error(nb_kde_pred, Ys_test)

print("Naive Bayes w/ KDE - Test Error:", nb_kde_test_err)

nb_gaussian = GaussianNB().fit(Xs_train, Ys_train)
nb_gaussian_pred = nb_gaussian.predict(Xs_test)
nb_gaussian_test_err = test_error(nb_gaussian_pred, Ys_test)

print("Naive Bayes w/ Gaussian - Test Error:", nb_gaussian_test_err)