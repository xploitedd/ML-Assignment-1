import numpy as np
from NaiveBayesKde import NaiveBayesKde
from SvmClassifier import SvmClassifier
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from Util import mcnemar_test, normal_test

train_data = np.loadtxt('TP1_train.tsv', delimiter='\t')
train_data = shuffle(train_data)

test_data = np.loadtxt('TP1_test.tsv', delimiter='\t')

Xs_train = train_data[:,0:-1]
Ys_train = train_data[:,-1]

Xs_test = test_data[:,0:-1]
Ys_test = test_data[:,-1]

# standardize the data
stds = np.std(Xs_train, axis=0)
means = np.mean(Xs_train, axis=0)

Xs_train = (Xs_train - means) / stds
Xs_test = (Xs_test - means) / stds

### Naive Bayes with Kernel Density Estimation ####

print("Optimizing KDE parameters...")

nb_kde = NaiveBayesKde()
nb_bw, nb_valid_err = nb_kde.optimize_bandwidth(Xs_train, Ys_train, 5, 0.02, 0.6, 0.02)

print("Naive Bayes w/ KDE - Best bandwidth:", nb_bw, "with validation error", nb_valid_err)

nb_kde.fit(Xs_train, Ys_train)
nb_kde_pred = nb_kde.predict(Xs_test, nb_bw)
nb_kde_test_err = 1 - accuracy_score(Ys_test, nb_kde_pred)

print("Naive Bayes w/ KDE - Test Error:", nb_kde_test_err)

#### Naive Bayes with a Gaussian Distribution ####

nb_gaussian = GaussianNB().fit(Xs_train, Ys_train)
nb_gaussian_pred = nb_gaussian.predict(Xs_test)
nb_gaussian_test_err = 1 - nb_gaussian.score(Xs_test, Ys_test)

print("\nNaive Bayes w/ Gaussian - Test Error:", nb_gaussian_test_err)

#### Support Vector Machine classifier with the Radial Basis Function kernel ####

print("\nOptimizing SVM parameters...")

svc = SvmClassifier()
svc_c, svc_g, svc_valid_err = svc.optimize_parameters(Xs_train, Ys_train, 5, [0.2, 6, 0.2], [1, 1000, 100])

print("SVM - Best C:               ", svc_c)
print("SVM - Best Gamma:           ", svc_g)
print("SVM - Best validation error:", svc_valid_err)

svc.fit(Xs_train, Ys_train, svc_g, svc_c)
svc_pred = svc.predict(Xs_test)
svc_test_err = 1 - svc.score(Xs_test, Ys_test)

print("SVM - Test Error:           ", svc_test_err)

#### Classifier comparison ####

nb_kde_errors, nb_kde_ntest = normal_test(Ys_test, nb_kde_pred)
nb_gaussian_errors, nb_gaussian_ntest = normal_test(Ys_test, nb_gaussian_pred)
svc_errors, svc_ntest = normal_test(Ys_test, svc_pred)

print("\n---- Normal Test ----")
print("Naive Bayes with KDE:     ", nb_kde_errors, "+-", nb_kde_ntest)
print("Naive Bayes with Gaussian:", nb_gaussian_errors, "+-", nb_gaussian_ntest)
print("SVM:                      ", svc_errors, "+-", svc_ntest)

nb_kde_vs_nb_gaussian = mcnemar_test(Ys_test, nb_kde_pred, nb_gaussian_pred)
nb_kde_vs_svc = mcnemar_test(Ys_test, nb_kde_pred, svc_pred) 
nb_gaussian_vs_svc = mcnemar_test(Ys_test, nb_gaussian_pred, svc_pred)

print("\n---- McNemar's Test ----")
print("Naive Bayes with KDE vs Naive Bayes with Gaussian:  ", nb_kde_vs_nb_gaussian)
print("Naive Bayes with KDE vs Support Vector Machine:     ", nb_kde_vs_svc)
print("Naive Bayes with Gaussian vs Support Vector Machine:", nb_gaussian_vs_svc)

nb_kde.plot_errors()
svc.plot_errors()
