#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis on Twitter Data using Bag of Words (BoW) and Logistic Regression
# 

# ## Introduction
# This notebook performs sentiment analysis on Twitter data using the Bag of Words approach with logistic regression. It aims to predict the sentiment (positive, negative, neutral) of tweets.

# 
# ## Setup
# Import the necessary libraries:
# - numpy: for numerical operations
# - pandas: for data manipulation and analysis
# - os: for interacting with the operating system

# In[ ]:


import numpy as np
import pandas as pd
import os


# In[2]:


train = pd.read_csv(r"C:\Users\VAGDEVI\Downloads\twitter_training.csv")
train


# In[3]:


train.columns=['id','information','type','text']
train


# In[4]:


val = pd.read_csv(r"C:\Users\VAGDEVI\Downloads\twitter_validation.csv")
val


# In[5]:


val.columns=['id','information','type','text']
val.head()


# ## Data Preprocessing
# Preprocess the text data by:
# - Converting the text to lowercase using the `lower` method.
# - Removing special characters and symbols using regular expressions (`re` library).

# In[7]:


import re #Regular expressions
#Text transformation
train["lower"]=train.text.str.lower() #lowercase
train["lower"]=[str(data) for data in train.lower] #converting all to string
train["lower"]=train.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x)) #regex
val["lower"]=val.text.str.lower() #lowercase
val["lower"]=[str(data) for data in val.lower] #converting all to string
val["lower"]=val.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x)) #regex


# In[8]:


train.head()


# ## Tokenization
# Tokenize the preprocessed text data into individual words using the `word_tokenize` function from the `nltk` library.

# In[9]:



import nltk
from nltk import word_tokenize
nltk.download('stopwords')
#Text splitting
tokens_text = [word_tokenize(str(word)) for word in train.lower]
#Unique word counter
tokens_counter = [item for sublist in tokens_text for item in sublist]
print("Number of tokens: ", len(set(tokens_counter)))


# ## Stopword Removal
# Remove English stopwords from the tokenized text data using the stopwords from the `nltk` corpus.

# In[10]:


#Choosing english stopwords
stopwords_nltk = nltk.corpus.stopwords
stop_words = stopwords_nltk.words('english')


# ## Bag of Words Representation
# Create a Bag of Words (BoW) representation of the tokenized text data using the `CountVectorizer` class from scikit-learn. Set the tokenizer to `word_tokenize` and specify the n-gram range as `(1, 1)` for analyzing individual words.

# In[11]:


from sklearn.feature_extraction.text import CountVectorizer #Data transformation
#Initial Bag of Words
bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    stop_words=stop_words, #English Stopwords
    ngram_range=(1, 1) #analysis of one word
)


# ## Train-Test Split
# Split the training data into training and testing sets using the `train_test_split` function from scikit-learn. We split 80% for training and 20% for testing.

# In[27]:


from sklearn.model_selection import train_test_split #Data testing
from sklearn.linear_model import LogisticRegression #Prediction Model
#Train - Test splitting
reviews_train, reviews_test = train_test_split(train, test_size=0.2, random_state=0)


# In[13]:


#Creation of encoding related to train dataset
X_train_bow = bow_counts.fit_transform(reviews_train.lower)
#Transformation of test dataset with train encoding
X_test_bow = bow_counts.transform(reviews_test.lower)


# In[14]:


X_test_bow


# In[15]:


#Labels for train and test encoding
y_train_bow = reviews_train['type']
y_test_bow = reviews_test['type']


# In[16]:


#Total of registers per category
y_test_bow.value_counts() / y_test_bow.shape[0]


# # Logistic Regression
# 
# The logistic regression model assumes a linear relationship between the independent variables and the log-odds of the dependent variable. The log-odds, also known as the logit function, is transformed using the sigmoid or logistic function, which maps any real-valued input to a value between 0 and 1. This sigmoid function allows us to interpret the output of the logistic regression model as probabilities.

# ![Sigmoid_graph.jpg](attachment:Sigmoid_graph.jpg)

# In[17]:


from sklearn.linear_model import LogisticRegression #Prediction Model
from sklearn.metrics import accuracy_score #Comparison between real and predicted
# Logistic regression
model1 = LogisticRegression(C=1, solver="liblinear",max_iter=200)
model1.fit(X_train_bow, y_train_bow)
# Prediction
test_pred = model1.predict(X_test_bow)
print("Accuracy: ", accuracy_score(y_test_bow, test_pred) * 100)


# In[18]:


#Validation data
X_val_bow = bow_counts.transform(val.lower)
y_val_bow = val['type']


# In[19]:


X_val_bow


# In[20]:


Val_res = model1.predict(X_val_bow)
print("Accuracy: ", accuracy_score(y_val_bow, Val_res) * 100)


# In[21]:


#n-gram of 4 words
bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    ngram_range=(1,4)
)
#Data labeling
X_train_bow = bow_counts.fit_transform(reviews_train.lower)
X_test_bow = bow_counts.transform(reviews_test.lower)
X_val_bow = bow_counts.transform(val.lower)


# In[22]:


X_train_bow


# In[23]:


model2 = LogisticRegression(C=0.9, solver="liblinear",max_iter=1500)
# Logistic regression
model2.fit(X_train_bow, y_train_bow)
# Prediction
test_pred_2 = model2.predict(X_test_bow)
print("Accuracy: ", accuracy_score(y_test_bow, test_pred_2) * 100)


# In[25]:


y_val_bow = val['type']
Val_pred_2 = model2.predict(X_val_bow)
print("Accuracy: ", accuracy_score(y_val_bow, Val_pred_2) * 100)


# ## Conclusion
# The notebook demonstrates the application of the Bag of Words approach and logistic regression for sentiment analysis on Twitter data. It provides insights into the accuracy of the model in predicting sentiment labels.

# In[ ]:




