#PROJECT 8: Fitting a support vector machine to the data ------
'''
Predict the sentiment (positive or negative) of a single sentence taken from a review of a movie, restaurant, or product.
The data set consists of 3000 labeled sentences, which we divide into a training set of size 2500 and a test set of size 500.
We have already used a logistic regression classifier. Now, we will use a support vector machine.
'''
import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

#Read in the data set:
with open("C:/Users/dubey/Downloads/data (1)/data/full_set.txt") as f:
     content = f.readlines()
     print(content)

#---PREPROCESSING THE TEXT DATA---
#Remove leading and trailing white space--
content = [x.strip() for x in content]
print("After remove leading and trailing white space------\n",content)
#Separate the sentences from the labels:
sentences =[x.split("\t")[0] for x in content]
labels = [x.split("\t")[1] for x in content]
#Transform the labels from '0 versus 1' to '-1 versus 1':
y = np.array(labels,dtype='int8')
y = 2*y - 1
'''
To transform this prediction problem into an linear classification, need to preprocess the 
text data. Do four transformations: 
o Remove punctuation and numbers. 
o Transform all words to lower-case. 
o Remove stop words. 
o Convert the sentences into vectors, using a bag-of-words representation. 
'''
#"full_remove" takes a string x and a list of characters
#"removal_list" and returns with all the characters in removal_list replaced by ' ' .
def full_remove(x,removal_list):
    for w in removal_list:
        x= x.replace(w, '')
        return x

digits = [str(x) for x in range(10)]
digit_less = [full_remove(x,digits) for x in sentences]
punc_less = [full_remove(x,list(string.punctuation)) for x in digit_less]
sents_lower = [x.lower() for x in punc_less]

#Stopwords -- ('a', 'an', 'the', 'i', 'he', 'she', 'they', 'to', 'of', 'it', from' )

stop_set = set(['a', 'an', 'the', 'i', 'he', 'she', 'they', 'to', 'of','is', 'it', 'from'])
#Remove stop words:
sents_split = [x.split() for x in sents_lower]
sents_processed = [" ".join(list(filter(lambda a: a not in stop_set, x))) for x in sents_split]
print("\nLet us look at the sentences: --------\n", sents_processed[0:10])

#Transform to bag of words representation....
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=4500)
data_features = vectorizer.fit_transform(sents_processed)
#Append '1' to end of each vector.
data_mat = data_features.toarray()
'''
#Training / test split - Split the data into a training set of 2500 
sentences and a test set of 500 sentences (of which 250 are positive and 250 
negative). 
# Split the data into testing and training sets
'''
np.random.seed(0)
test_inds = np.append(np.random.choice((np.where(y==-1))[0],250,replace=False),np.random.choice((np.where(y==1))[0], 250, replace=False))
train_inds = list(set(range(len(labels))) - set(test_inds))
train_data = data_mat[train_inds,]
train_labels = y[train_inds]

test_data = data_mat[test_inds,]
test_labels = y[test_inds]

print('Train Data :',train_data.shape)
print('Test Data :',test_data.shape)
'''
Fitting a support vector machine to the data ----
In support vector machines, we are given a set of examples  ( 1, 1),…,(  ,  )  and we want to find a weight 
vector   ∈ℝ   that solves the following optimization problem: 
scikit-learn provides an SVM solver that we will use. The following routine takes the constant "C" as an input 
(from the above optimization problem) and returns the training and test error of the resulting SVM model. It 
is invoked as follows: 
 "training_error, test_error = fit_classifier(C)" 
 The default value for parameter "C" is 1.0.
'''

def fit_classifier(C_value=1.0):
    clf = svm.LinearSVC(C=C_value, loss='hinge')
    clf.fit(train_data,train_labels)
# Get predictions on training data
    train_preds = clf.predict(train_data)
    train_error = float(np.sum((train_preds > 0.0) != (train_labels > 0.0)))/len(train_labels)
# Get predictions on test data
    test_preds = clf.predict(test_data)
    test_error = float(np.sum((test_preds > 0.0) != (test_labels > 0.0)))/len(test_labels)
    return train_error, test_error
cvals = [0.01,0.1,1.0,10.0,100.0,1000.0,10000.0]
for c in cvals:
    train_error, test_error = fit_classifier(c)
    print ("Error rate for C = %0.2f: train %0.3f test %0.3f" % (c, train_error, test_error))
'''
3. Evaluating C by k-fold cross-validation-------
 As we can see, the choice of "C" has a very significant effect on the performance of the SVM 
classifier. We were able to assess this because we have a separate test set. In general, however, this 
is a luxury we won't possess. 
 A reasonable way to estimate the error associated with a specific value of "C" is by k-fold cross 
validation: 
o Partition the training set into "k" equal-sized sized subsets. 
o For i=1,2,...,k, train a classifier with parameter C.. 
 Average the errors: "(e_1 + ... + e_k)/k" 
 The following procedure, cross_validation_error, does exactly this. It takes as input: 
o the training set "x,y" 
o the value of "C" to be evaluated 
o the integer "k" 
 It returns the estimated error of the classifier for that particular setting of "C".
'''

# 3. Evaluating C by k-fold cross-validation
def cross_validation_error(x, y, C_value, k):
    n = len(y)
    # Randomly shuffle indices
    indices = np.random.permutation(n)
    # Initialize error
    err = 0.0
    # Iterate over partitions
    for i in range(k):
        # Partition indices
        test_indices = indices[int(i * (n / k)):int((i + 1) * (n / k) - 1)]
        train_indices = np.setdiff1d(indices, test_indices)
        # Train classifier with parameter c
        clf = svm.LinearSVC(C=C_value, loss='hinge')
        clf.fit(x[train_indices], y[train_indices])
        # Get predictions on test partition
        preds = clf.predict(x[test_indices])
        # Compute error
        err += float(np.sum((preds > 0.0) != (y[test_indices] >0.0))) / len(test_indices)
    return err / k


# Let us print out the errors or different vaues of k...........
for k in range(2, 10):
    print("cross_validation_error: ")
    print(cross_validation_error(train_data, train_labels, 1.0, k))



#Calculate the Model Accuracy , MAE, MSE , RMSE and R SQUARE SCORE-----
from sklearn import metrics
clf = svm.LinearSVC(C=1.0, loss='hinge')
clf.fit(train_data, train_labels)
# Get predictions on the test set
test_preds = clf.predict(test_data)

accuracy = metrics.accuracy_score(test_labels, test_preds)
print("Accuracy Score: ", accuracy)

mae = metrics.mean_absolute_error(test_labels, test_preds)
print("Mean Absolute Error = ", mae)

mse = metrics.mean_squared_error(test_labels, test_preds)
print("Mean Squared Error: ", mse)

rmse = mse ** 0.5
print("Root Mean Squared Error: ", rmse)

print("R Square Score: ", metrics.r2_score(test_labels, test_preds))



'''
Accuracy Score:  0.852
Mean Absolute Error =  0.296
Mean Squared Error:  0.592
Root Mean Squared Error:  0.7694153624668538
R Square Score:  0.40800000000000003
'''