# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:12:25 2024

@author: shubh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


#importing data set

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#cleaning the data 
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords  ## include the stopwords after we downloaded them.. stop words like the is etc.
from nltk.stem.porter import PorterStemmer 
corpus = []  # this will contain cleaned reviews
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) # thats the colum one where it has review
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
    
    

#creating the bag of words model    

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer() # if put max number or most frequeent values here then we can elimenate the unnacceary items
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

#Spliting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Naive Bayes model on training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

# confusion mattrix 

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))







    