import numpy as np
from NaiveBayesKde import NaiveBayesKde
from sklearn.utils import shuffle

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

nb_kde = NaiveBayesKde()
best_bw, best_valid_err = nb_kde.fit(Xs_train, Ys_train, 5, 0.02, 0.61, 0.02)

print("Best bandwidth is", best_bw, "with validation error", best_valid_err)

predictions = nb_kde.predict(Xs_test)
test_error = nb_kde.test_error(predictions, Ys_test)

print("Naive Bayes w/ KDE - Test Error:", test_error)