# ---PROJECT 4: SENTIMENT ANALYSIS USING LOGISTIC REGRESSION ---
'''
#About the dataset----
 The "sentiment" dataset consists of 3000 sentences which come from reviews on "imdb.com",
"amazon.com", and "yelp.com". Each sentence is labeled according to whether it comes from a
positive review or negative review.
 The data set consists of 3000 sentences, each labeled '1' (if it came from a positive review) or '0' (if it
came from a negative review). To be consistent with our notation from lecture, we will change the
negative review label to '-1'.
 We will use logistic regression to learn a classifier from this data.
'''

import string
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.rc('xtick',labelsize=14)
matplotlib.rc('ytick',labelsize=14)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings('ignore')

#Read the dataset:
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

'''
Bag of words 
 In order to use linear classifiers on our data set, we need to transform our textual data into numeric 
data. The classical way to do this is known as the bag of words representation. 
 In this representation, each word is thought of as corresponding to a number in "{1, 2, ..., V}" where 
"V" is the size of our vocabulary. And each sentence is represented as a V-dimensional vector    , 
where      is the number of times that word     occurs in the sentence. 
 To do this transformation, we will make use of the "CountVectorizer" class in "scikit-learn". We will 
cap the number of features at 4500, meaning a word will make it into our vocabulary only if it is one 
of the 4500 most common words in the corpus. This is often a useful step as it can weed out spelling 
mistakes and words which occur too infrequently to be useful. 
 Finally, we will also append a '1' to the end of each vector to allow our linear classifier to learn a bias 
term.
'''

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
2. Fitting a logistic regression model to the training data-----
 
 We could implement our own logistic regression solver using stochastic gradient descent, 
but fortunately, there is already one built into "scikit-learn". 
 Due to the randomness in the SGD procedure, different runs can yield slightly different 
solutions (and thus different error values).
'''
#Fit logistic classifier on training data:
clf = SGDClassifier(loss="log_loss")
clf.fit(train_data, train_labels)
# Pull out the parameters (w,b) of the logistic regression model
w= clf.coef_[0,:]
b= clf.intercept_
#Get predictions on training and test data
preds_train = clf.predict(train_data)
preds_test = clf.predict(test_data)
#Compute Errors:
errs_train = np.sum((preds_train>0.0) != (train_labels >0.0))
errs_test = np.sum((preds_test>0.0) != (test_labels >0.0))
print("Training Error:", float(errs_train)/len(train_labels))
print("Test Error:", float(errs_test)/len(test_labels))

'''
3. Analyzing the margin-----
 
 The logistic regression model produces not just classifications but also conditional 
probability estimates. 
 We will say that "x" has margin "gamma" if (according to the logistic regression model) 
"Pr(y=1|x) > (1/2)+gamma" or "Pr(y=1|x) < (1/2)-gamma". The following function 
margin_counts takes as input as the classifier ("clf", computed earlier), the test set 
("test_data"), and a value of "gamma", and computes how many points in the test set have 
margin of at least "gamma
'''
# Return number of test points for which Pr(y=1) lies in [0, 0.5 - gamma) or (0.5 + gamma, 1]


def margin_counts(clf, test_data, gamma):
# Compute probability on each test point
   preds = clf.predict_proba(test_data)[:, 1]
# Find data points for which prediction is at least gamma away from 0.5
   margin_inds = np.where((preds > (0.5 + gamma)) | (preds < (0.5-gamma)))[0]
   return float(len(margin_inds))

# Let us visualize the test set's distribution of margin values.
gammas = np.arange(0, 0.5, 0.01)
f = np.vectorize(lambda g: margin_counts(clf, test_data, g))
plt.plot(gammas, f(gammas) / 500.0, linewidth=2, color='green')
plt.xlabel('Margin', fontsize=14)
plt.ylabel('Fraction of points above margin', fontsize=14)
plt.show()

'''
We investigate a natural question: 
"Are points "x" with larger margin more likely to be classified correctly? 
To address this, we define a function margin_errors that computes the fraction of points with margin at least "gamma" 
that are misclassified. 
Return error of predictions that lie in intervals [0, 0.5 - gamma) and (0.5 + gamma, 1] 
'''
def margin_errors(clf, test_data, test_labels, gamma):
    # Compute probability on each test point
    preds = clf.predict_proba(test_data)[:,1]
    # Find data points for which prediction is at least gamma away from 0.5
    margin_inds = np.where((preds > (0.5+gamma)) | (preds < (0.5 - gamma)))[0]
    # Compute error on those data points.
    num_errors = np.sum((preds[margin_inds] > 0.5) != (test_labels[margin_inds] > 0.0))
    return float(num_errors)/len(margin_inds)
#Let us visualize the relationship between margin and error rate.
#Create grid of gamma values
gammas=  np.arange(0,0.5,0.01)
#Compute margin _errors on test data for each value of g
f = np.vectorize(lambda g: margin_errors(clf,test_data,test_labels,g))
#Plot the result
plt.plot(gammas,f(gammas),linewidth=2)
plt.ylabel('Error Rate', fontsize= 14)
plt.xlabel('Margin',fontsize=14)
plt.show()
'''
4. Words with large influence 
 Finally, we attempt to partially interpret the logistic regression model. 
 Which words are most important in deciding whether a sentence is positive? As a first approximation 
to this, we simply take the words whose coefficients in "w" have the largest positive values. 
 Likewise, we look at the words whose coefficients in "w" have the most negative values, and we 
think of these as influential in negative predictions.
'''
# Convert vocabulary into a list:
vocab = np.array([z[0] for z in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1])])
# Get indices of sorting w
inds = np.argsort(w)
# Words with large negative values
neg_inds = inds[0:50]
print("Highly negative words: ")
print([str(x) for x in list(vocab[neg_inds])])
# Words with large positive values
pos_inds = inds[-49:-1]
print("Highly positive words: ")
print([str(x) for x in list(vocab[pos_inds])])

#Calculate the Model Accuracy , MAE, MSE , RMSE and R SQUARE SCORE-----
from sklearn import metrics

#Fit logistic classifier on training data:
clf = SGDClassifier(loss="log_loss")
clf.fit(train_data, train_labels)

#Get predictions on test data
preds_test = clf.predict(test_data)

accuracy = metrics.accuracy_score(test_labels, preds_test)
print("Accuracy Score: ", accuracy)

mae = metrics.mean_absolute_error(test_labels, preds_test)
print("Mean Absolute Error = ", mae)

mse = metrics.mean_squared_error(test_labels, preds_test)
print("Mean Squared Error: ", mse)

rmse = mse ** 0.5
print("Root Mean Squared Error: ", rmse)

print("R Square Score: ", metrics.r2_score(test_labels, preds_test))

'''
Accuracy Score:  0.856
Mean Absolute Error =  0.288
Mean Squared Error:  0.576
Root Mean Squared Error:  0.758946638440411
R Square Score:  0.42400000000000004
'''