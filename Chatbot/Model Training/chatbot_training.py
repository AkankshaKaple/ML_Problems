
# coding: utf-8

# ## <b>AI based Chatbot</b>

# ## Problem Statement:
# An AI based chatbot where we have to predict the answer of the user's asked question.
# ## Dataset:
# Here we are using Bridgelabz website chatbot dataset
# 
# ## Solution:
# Here, we will use NLP for text processing and for prediction Machine learning SVM algorithm.

# ### What you may learn here?
# This notebook is a detailed investigation on AI based chatbot, how it will work, how we will pre-process the chatbot dataset and how we predict the answer from user's asked question.
# 

# ## Step 1: Import all the required libraries 
# 
# * __NLTK__ : The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing for English written in the Python programming language
# * __Pandas__ : In computer programming, pandas is a software library written for the Python programming language for data manipulation and analysis and storing in a proper way. In particular, it offers data structures and operations for manipulating numerical tables and time series
# * __Numpy__ : NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
# * __Sklearn__ : Scikit-learn (formerly scikits.learn) is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy. The library is built upon the SciPy (Scientific Python) that must be installed before you can use scikit-learn.
# * __Pickle__ : Python pickle module is used for serializing and de-serializing a Python object structure. Pickling is a way to convert a python object (list, dict, etc.) into a character stream. The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.

# In[1]:


#Loading libraries 
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import sklearn
import re


# ## Step 2 : Read all the required data and combine it

# In[3]:


#loading data
try:
    faq = pd.read_csv('Data/chatbot_faq.csv')
    greeting = pd.read_csv('Data/Greetings.csv')
except (FileNotFoundError, IOError):
    print("Wrong file or file path")   


# Now we will concat the faq and greeting in one dataset

# In[4]:


data = pd.concat([faq, greeting], ignore_index=True)


# After we read the data, we can look at the data using:
# 

# In[5]:


print(data.head())


# In[6]:


print ('The dataset has {0} rows and {1} columns'.format(data.shape[0],data.shape[1]))


# Alternatively, you can also check the data set information using the info() command.
# 
# 

# Let's check if the data set has any missing values.
# 
# 

# ## Step 3: Data Exploration
# 

# ## Step 4: Data pre-processing
# 

# In this stage, we'll deal with outlier values, encode variables, impute missing values, and take every possible initiative which can remove inconsistencies from the data set. Let's remove that:

# * First we'll tokenzie each word from the dataset.
# * After we tokenize, we will start cleaning up the tokens by Lemmatizing. Lemmatizing is the process of converting a word into its root form. 
# - __Tokenizing__ : This breaks up the strings into a list of words or pieces based on a specified pattern using Regular Expressions aka RegEx. 
# - eg : white brown fox = ‘white’, ‘brown’,’fox’
# - __Lemmatizing__ : Lemmatizing is the process of converting a word into its root form.
# - e.g., "Playing", "Played" = "play".
# 
# In cleanup() function, we are first tokenizing the sentence (seperating each word in sentence) and then steeming (converting a word into its root form) and at the end combine all the words to form a sentence.

# In[7]:


import re
class Data_Cleanig:
    def data_cleanup(self, sentence):
        TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
        cleaned_text = re.sub(TEXT_CLEANING_RE, ' ', str(sentence).lower()).strip()
        word_tok = nltk.word_tokenize(cleaned_text)
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(w) for w in word_tok]
        return ' '.join(lemmatized_words)


#  Pass each question to the cleaning funtion defined above

# In[8]:


cleaning = Data_Cleanig()
questions_cleaned = []
questions = data['Question'].values
for question in questions:
    questions_cleaned.append(cleaning.data_cleanup(question))


# The sentence __"How long is the fellowship program?"__ converted to __"how long is the fellow program ?"__

# In[9]:


class Preprocessing():       
    # Vectorization for training
    def vectorize(self, clean_questions):
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=1, stop_words='english')  
        vectorizer.fit(clean_questions)
        transformed_X_csr = vectorizer.transform(clean_questions)
        transformed_X = transformed_X_csr.A # csr_matrix to numpy matrix  
        return transformed_X, vectorizer

    # Vectorization for input query
    def query(self, clean_usr_msg, usr, vectorizer):
        t_usr_array= None
        try:
            t_usr = vectorizer.transform([clean_usr_msg])
            t_usr_array = t_usr.toarray()
        except Exception as e:
            print(e)
            return "Could not follow your question [" + usr + "], Try again"

        return t_usr_array


# In[10]:


le = LabelEncoder()
preprocessing = Preprocessing()
X, vectorizer = preprocessing.vectorize(questions_cleaned)
y = data['Class'].values.tolist()
y = le.fit_transform(y)


# ## Step 5: Split the data into train and test set
# - Now our data is ready to feed to the program. But here we'll split the data into train and test dataset so that after training the model we can test the model on the test dataset and find out how accurate are its predictions.
# - Here we are splitting the data so that the training dataset contains 80% of the data and the test dataset contains 20% of the total data.
# - Here we are using the train_test_split method from the sklearn library. We'll train our model on x_train and y_train, test it on x_test and y_test.
# 
# - test_size: Here we specify the size we want for our test dataset.
# - random_state: When we use a random number generator for number or sequence generation, we give a starting number (AKA seed). When we provide the same seed, every time it’ll generate the same sequence as the first one. That’s why to keep the same random values every time, we give seed as random_state in train_test_split().

# In[11]:


#split the dataset into x and y
x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        X, y, test_size=0.25,random_state=42)


# We can't see the values of an entire matrix, but by looking at its shape, we decide if we going in the right direction or not. By using ".shape" we can see shape of a matrix and it will also helpful in debugging.

# In[12]:


print(x_data_train.shape, x_data_test.shape, y_data_train.shape, y_data_test.shape)


# ## Step 6: Train the Model using SVM

# In[13]:


#Using sklearn linear regression model
model = sklearn.svm.SVC(kernel='linear')
model.fit(x_data_train, y_data_train)
print("SVC:", model.score(x_data_test, y_data_test))


# ## Step 7: Save the model in a pickle file
# 
# As logistic_model_cv gave us the highest accuracy we'll go with it and save it to pickle file.
# We save our model to pickle files so that when we want to perform predictions on unseen data, we don't have to train our model again. Any object in python can be pickled so that it can be saved on disk. What pickle does is that it “serializes” the object first before writing it to file. Pickling is a way to convert a python object (list, dict, etc.) into a character stream. 

# In[15]:


with open('Model/chatbot_model.pkl','wb') as f:
        pickle.dump(cleaning, f)
        pickle.dump(preprocessing, f)
        pickle.dump(vectorizer,f)
        pickle.dump(model, f)


# ## Summary

# The project has been created to help people understand the complete process of machine learning / data science modeling. These steps ensure that you won't miss out any information in the data set and would also help another person understand your work.
