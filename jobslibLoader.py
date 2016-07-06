from sklearn.externals import joblib
from sklearn import datasets,svm,metrics

clf = joblib.load('svmClf.pkl')
digits = datasets.load_digits()
n_samples = len(digits.data)
clf.fit(digits.data[:n_samples/2], digits.target[:n_samples/2])

# Predict using the second half of the data
expected = digits.target[n_samples/2:]
predicted = clf.predict(digits.data[n_samples/2:])

print 'Confusion matrix: \n{0}'.format(metrics.confusion_matrix(expected,predicted))

