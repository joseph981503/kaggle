from __future__ import division
import numpy as np
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
from sklearn.ensemble import RandomForestRegressor

SEED = 30


def load_data(filename, use_labels=True):

    data = np.loadtxt(open(filename), delimiter=',' , usecols=range(1, 9), skiprows=1)
    if use_labels:
        labels = np.loadtxt(open(filename), delimiter=',' , usecols=[0], skiprows=1)
    else:
        labels = np.zeros(data.shape[0])
    return labels, data


def save_results(predictions, filename):

    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))


def main():

    model = RandomForestRegressor(random_state=0)  # the classifier we'll use

# === load data in memory === #
    y, X = load_data('train.csv')
    y_test, X_test = load_data('test.csv', use_labels=False)

    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    X_test = encoder.transform(X_test)

# === training & metrics === #
    mean_auc = 0.0
    n = 10  # repeat the CV procedure 10 times to get more precise results
    for i in range(n):
    # for each iteration, randomly hold out 20% of the data as CV set
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=.40, random_state=i*SEED)


    # train model and make predictions
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:, 1]

    # compute AUC metric for this CV fold
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
        roc_auc = metrics.auc(fpr, tpr)
        print ("AUC (fold %d/%d): %f" % (i + 1, n, roc_auc))
        mean_auc += roc_auc

        print ("Mean AUC: %f" % (mean_auc/n))
    X, y= make_regression(random_state=0, shuffle=False)
    model.fit(X, y)
    preds = model.predict_proba(X_test)[:, 1]
    save_results(preds, "output1" + ".csv")

if __name__ == '__main__':
    main()
