# This is an scikit-learn example on digit classification

from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
from sklearn.externals import joblib

# Datasets are a dictionary type variable
digits = datasets.load_digits()

# Show keys in the dataset
# .data -> n_samples, n_features array. Use to train the classifier
# .target -> index corresponds to data index. Gives ground truth, i.e. this sample should be in which class
# .DESCR -> Backgrd of the dataset
print digits.keys()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 3 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# pylab.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images,digits.target))
for index, (image,label) in enumerate(images_and_labels[:4]):
	# subplot(nrows,ncols,plot_number)
	# plot_number starts from 1
	plt.subplot(2,4,index+1)
	plt.axis('off')
	plt.imshow(image,cmap=plt.cm.gray_r, interpolation='nearest')
	plt.title('Training: %i' % label)

clf = svm.SVC(gamma=0.001)

# train using the first half of the data. 
n_samples = len(digits.data)
clf.fit(digits.data[:n_samples/2], digits.target[:n_samples/2])

# Predict using the second half of the data
expected = digits.target[n_samples/2:]
predicted = clf.predict(digits.data[n_samples/2:])

print 'classification report for classifier {0}\n{1}'.format(clf,metrics.classification_report(expected,predicted))

print 'Confusion matrix: \n{0}'.format(metrics.confusion_matrix(expected,predicted))

images_and_predictions = list(zip(digits.images[n_samples/2:], predicted))
for index, (image,prediction) in enumerate(images_and_predictions[:4]):
	plt.subplot(2,4,index+5)
	plt.axis('off')
	plt.imshow(image,cmap=plt.cm.gray_r, interpolation='nearest')
	plt.title('Prediction {0}'.format(prediction))

#plt.show()

# save trained classifier to pickle file
# joblib.dump returns a list of filenames. Each individual numpy array contained in the clf object is serialized as separate file in the folder. All files are required in the same fodler when reloading the model with joblib.load
joblib.dump(clf,'svmClf.pkl')

