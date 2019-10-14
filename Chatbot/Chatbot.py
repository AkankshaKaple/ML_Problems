
# coding: utf-8

# ## Introduction

# For freshers, projects are the best way to highlight their data science knowledge. In fact, not just freshers, up to mid-level experienced professionals can keep their resumes updated with new, interesting projects. After all, they don't come easy. It takes a lot of time to create a project which can truly showcase the depth and breadth of your knowledge.
# 
# I hope this project will help you gain much needed knowledge and help your resume get shortlisted faster. This project shows all the steps (from scratch) taken to solve a Machine Learning problem. For your understanding, I've taken a simple yet challenging data set where you can engineer features at your discretion as well.
# 
# This project is most suitable for people who have a basic understanding of python and Machine Learning. Even if you are absolutely new to it, give it a try. And ask questions in Comments below. R users can refer to this equivalent R script and follow the explanation given below.

# ## Table of Contents
# 

# 1. Process of Machine Learning Predictions
# 2. Chatbot Data Set
#     - Understand the problem
#     - Hypothesis Generation
#     - Get Data
#     - Data Exploration
#     - Data Pre-Processing
#     - Feature Engineering - Create new features
#     - Model Training - XGBoost, Neural Network, Lasso
#     - Model Evaluation

# ## Process of Machine Learning Predictions
# 

# “Keep tormenting data until it starts revealing its hidden secrets.” Yes, it can be done but there's a way around it. Making predictions using Machine Learning isn't just about grabbing the data and feeding it to algorithms. The algorithm might spit out some prediction but that's not what you are aiming for. The difference between good data science professionals and naive data science aspirants is that the former set follows this process religiously. The process is as follows:
# 

# ### 1. Understand the problem: 
# Before getting the data, we need to understand the problem we are trying to solve. If you know the domain, think of which factors could play an epic role in solving the problem. If you don't know the domain, read about it. 
# 
# ### 2. Hypothesis Generation: 
# This is quite important, yet it is often forgotten. In simple words, hypothesis generation refers to creating a set of features which could influence the target variable given a confidence interval ( taken as 95% all the time). We can do this before looking at the data to avoid biased thoughts. This step often helps in creating new features. 
# 
# ### 3. Get Data: 
# Now, we download the data and look at it. Determine which features are available and which aren't, how many features we generated in hypothesis generation hit the mark, and which ones could be created. Answering these questions will set us on the right track. 
# 
# ### 4. Data Exploration: 
# We can't determine everything by just looking at the data. We need to dig deeper. This step helps us understand the nature of variables (skewed, missing, zero variance feature) so that they can be treated properly. It involves creating charts, graphs (univariate and bivariate analysis), and cross-tables to understand the behavior of 
# features.
# 
# ### 5. Data Preprocessing: 
# Here, we impute missing values and clean string variables (remove space, irregular tabs, data time format) and anything that shouldn't be there. This step is usually followed along with the data exploration stage. 
# 
# ### 6. Feature Engineering: 
# Now, we create and add new features to the data set. Most of the ideas for these features come during the hypothesis generation stage. 
# 
# ### 7. Model Training: 
# Using a suitable algorithm, we train the model on the given data set. 
# 
# ### 8. Model Evaluation: 
# 
# Once the model is trained, we evaluate the model's performance using a suitable error metric. Here, we also look for variable importance, i.e., which variables have proved to be significant in determining the target variable. And, accordingly we can shortlist the best variables and train the model again. 
# 
# ### 9. Model Testing: 
# Finally, we test the model on the unseen data (test data) set.
# 

# We'll follow this process in the project to arrive at our final predictions. Let's get started.
# 
# 

# ## 1.Understand the problem

# The data set for this project has been taken from Bridgelabz Chatbot product. As mentioned above, the data set is simple. This project aims at predicting answer based on user's input question. I believe this problem statement is quite self-explanatory and doesn't need more explanation. Hence, we move to the next step.
# 
# 

# ## 2. Hypothesis Generation
# 

# Well, this is going to be interesting. What factors can you think of right now which can influence chatbot question & answer class ? As you read this, I want you to write down your factors as well, then we can match them with the data set. Defining a hypothesis has two parts: Null Hypothesis (Ho) and Alternate Hypothesis(Ha). They can be understood as:
# 
# Ho - There exists no impact of a particular feature on the dependent variable. Ha - There exists a direct impact of a particular feature on the dependent variable.
# 
# Based on a decision criterion (say, 5% significance level), we always 'reject' or 'fail to reject' the null hypothesis in statistical parlance. Practically, while model building we look for probability (p) values. If p value < 0.05, we reject the null hypothesis. If p > 0.05, we fail to reject the null hypothesis. Some factors which I can think of that directly influence house prices are the following:
# 
# Intent of the user's asked question
# 

# ## 3. Get Data
# 

# You can download the data and load it in your python IDE.Yes, it's going to be one heck of a data exploration ride. But, we'll learn how to deal with so many variables. The target variable is question & answer class. As you can see the data set comprises numeric, categorical, and ordinal variables. Without further ado, let's start with hands-on coding.
# 
# 

# ## 4. Data Exploration
# 

# Data Exploration is the key to getting insights from data. Practitioners say a good data exploration strategy can solve even complicated problems in a few hours. A good data exploration strategy comprises the following:
# 
# 1. Univariate Analysis - It is used to visualize one variable in one plot. Examples: histogram, density plot, etc.
# 2. Bivariate Analysis - It is used to visualize two variables (x and y axis) in one plot. Examples: bar chart, line chart, area chart, etc.
# 3. Multivariate Analysis - As the name suggests, it is used to visualize more than two variables at once. Examples: stacked bar chart, dodged bar chart, etc.
# 4. Cross Tables -They are used to compare the behavior of two categorical variables (used in pivot tables as well).

# Let's load the necessary libraries and data and start coding.
# 
# 

# ### Import libraries and files
# 
# * __NLTK__ : The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing for English written in the Python programming language
# * __Pandas__ : In computer programming, pandas is a software library written for the Python programming language for data manipulation and analysis and storing in a proper way. In particular, it offers data structures and operations for manipulating numerical tables and time series
# * __Numpy__ : NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
# * __Sklearn__ : Scikit-learn (formerly scikits.learn) is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy. The library is built upon the SciPy (Scientific Python) that must be installed before you can use scikit-learn.
# * __Pickle__ : Python pickle module is used for serializing and de-serializing a Python object structure. Pickling is a way to convert a python object (list, dict, etc.) into a character stream. The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.

# In[72]:


#Loading libraries 
import pandas as pd
import numpy as np
import pickle
import operator
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics.pairwise import cosine_similarity
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


# In[73]:


#loading data

try:
    faq = pd.read_csv('data/chatbot/chatbot_faq.csv')
    greeting = pd.read_csv('data/chatbot/Greetings.csv')
except (FileNotFoundError, IOError):
    print("Wrong file or file path")   


# Now we will concat the faq and greeting in one dataset

# In[74]:


data = pd.concat([faq, greeting], ignore_index=True)


# After we read the data, we can look at the data using:
# 

# In[75]:


data.head()


# In[76]:


print ('The dataset has {0} rows and {1} columns'.format(data.shape[0],data.shape[1]))


# Alternatively, you can also check the data set information using the info() command.
# 
# 

# In[77]:


data.info()


# Let's check if the data set has any missing values.
# 
# 

# In[78]:


#check missing values
data.columns[data.isnull().any()]


# In[79]:


#missing value counts in each of these columns
miss = data.isnull().sum()/len(data)
miss = miss[miss > 0]
miss.sort_values(inplace=True)
miss


# There is no missing data. In some case if there will be some missing data then we have to check the percentage of missing values in there columns 
# 
# 

# ## 5. Data Pre-Processing
# 

# In this stage, we'll deal with outlier values, encode variables, impute missing values, and take every possible initiative which can remove inconsistencies from the data set. If you remember, we discovered that the variable GrLivArea has outlier values. Precisely, one point crossed the 4000 mark. Let's remove that:

# * First we'll tokenzie each word from the dataset.
# * After we tokenize, we will start cleaning up the tokens by Lemmatizing. Lemmatizing is the process of converting a word into its root form. 
# - __Tokenizing__ : This breaks up the strings into a list of words or pieces based on a specified pattern using Regular Expressions aka RegEx. 
# - eg : white brown fox = ‘white’, ‘brown’,’fox’
# - __Lemmatizing__ : Lemmatizing is the process of converting a word into its root form.
# - e.g., "Playing", "Played" = "play".
# 
# In cleanup() function, we are first tokenizing the sentence (seperating each word in sentence) and then steeming (converting a word into its root form) and at the end combine all the words to form a sentence.

# In[80]:


def cleanup( sentence):
    word_tok = nltk.word_tokenize(sentence)
    stemmer = LancasterStemmer()
    stemmed_words = [stemmer.stem(w) for w in word_tok]
    return ' '.join(stemmed_words)


#  Pass each question to the cleaning funtion defined above

# In[81]:


questions_cleaned = []
questions = data['Question'].values
for question in questions:
    questions_cleaned.append(cleanup(question))


# The sentence __"How long is the fellowship program?"__ converted to __"how long is the fellow program ?"__

# In[82]:


# Vectorization for training
def vectorize(clean_questions):
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')  
    vectorizer.fit(clean_questions)
    transformed_X_csr = vectorizer.transform(clean_questions)
    transformed_X = transformed_X_csr.A # csr_matrix to numpy matrix  
    return transformed_X, vectorizer

# Vectorization for input query
def query(clean_usr_msg, vectorizer):
    t_usr_array= None
    try:
        t_usr = vectorizer.transform([clean_usr_msg])
        t_usr_array = t_usr.toarray()
    except Exception as e:
        print(e)
        return "Could not follow your question [" + usr + "], Try again"

    return t_usr_array


# In[83]:


le = LabelEncoder()

X, vectorizer = vectorize(questions_cleaned)
Y = data['Class'].values.tolist()
Y = le.fit_transform(y)


# ## 6. Feature Engineering
# 

# There are no libraries or sets of functions you can use to engineer features. Well, there are some but not as effective. It's majorly a manual task but believe me, it's fun. Feature engineering requires domain knowledge and lots of creative ideas. The ideas for new features usually develop during the data exploration and hypothesis generation stages. The motive of feature engineering is to create new features which can help make predictions better.

# ## 7. Model Training and Evaluation
# 

# Split the Data Into Training and Test Subsets
# In this step we will split our dataset into training and testing subsets (in proportion 80/20%).
# 
# Training data set will be used for training of our linear model. Testing dataset will be used for validating of the model. All data from testing dataset will be new to model and we may check how accurate are model predictions.

# In[84]:


#split the dataset into x and y
x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        X, Y, test_size=0.25,random_state=42)


# In[85]:


#Using sklearn linear regression model

algorithms = {'LinearRegression()':'LinearRegression', 'linear_model.Lasso(alpha=0.1)' : 'Lasso LinearRegression', 
              'LassoCV()' : 'LassoCV LinearRegression', 'DecisionTreeRegressor()': 'DecisionTree', 'SVR()':'SVR'}
acc_train_test = []
r2_train_test = []
results =  {}
for key, value in algorithms.items():
        model = eval(key)
        reg = model.fit(x_data_train,y_data_train) # training the dataset
        y_pred_train =  reg.predict(x_data_train)  # predicting the results
        y_pred_test =  reg.predict(x_data_test)
        
        r2_train_test.append([value, (r2_score(y_data_train,y_pred_train),r2_score(y_data_test,y_pred_test)), 
                              (reg.score(x_data_train,y_data_train), reg.score(x_data_test,y_data_test))])
results.update({"data": r2_train_test})        
x = PrettyTable()
x.field_names = ["Algorithm", "R2Score (Train/Test)", "Accuracy (Train/Test)"]
r2score = results['data']
for val in range(0 , len(r2score)):
    r2_value = str(math.floor(r2score[val][2][0]*100)) + '/' + str(math.floor(r2score[val][2][1]*100))
    acc_value = str(math.floor(r2score[val][1][0]*100)) + '/' + str(math.floor(r2score[val][1][1]*100))
    x.add_row([r2score[val][0], r2_value, acc_value])
print(x)    


# ## Step 8: Save model to the pickle file

# In[86]:


with open('chatbot.pkl','wb') as f:
    pickle.dump(model,f)


# ## Step 9: Test the model
# 
# Enter your query in the and check the output.

# In[87]:


usr = 'What is fellowship'
cleaned_usr = cleanup(usr)
t_usr_array = query(cleaned_usr, vectorizer)
prediction = model.predict(t_usr_array)[0]
class_ = le.inverse_transform([prediction])[0]
questionset = data[data['Class']==class_]

cos_sims = []
for question in questionset['Question']:
    cleaned_question = cleanup(question)
    question_arr = query(cleaned_question, vectorizer)
    sims = cosine_similarity(question_arr, t_usr_array)
    cos_sims.append(sims)

if len(cos_sims) > 0:
    ind = cos_sims.index(max(cos_sims)) 
    print(data['Answer'][questionset.index[ind]])


# ## Summary

# The project has been created to help people understand the complete process of machine learning / data science modeling. These steps ensure that you won't miss out any information in the data set and would also help another person understand your work. I would like to thank the Kaggle community for sharing info on competition forums which helped me a lot in creating this tutorial.

# ## References

# <p>Hackerearth machine learning project <a href="https://www.hackerearth.com/practice/machine-learning/machine-learning-projects/python-project/tutorial/">here</a></p>
# 
# <p>Chatbot dataset <a href="http://bridgelabz.com">here</a></p>
# 
# <p>Machine learning code <a href="https://github.com/Deepakchawla/Machine-Learning-Problems">here</a></p>
