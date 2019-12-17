
# coding: utf-8

# ## <b>Art of Happiness Model Training</b>

# ## Problem Statement:
# The World Happiness Report is a landmark survey of the state of global happiness. The first report was published in 2012, the second in 2013, the third in 2015, and the fourth in the 2016 Update. The World Happiness 2017, which ranks 155 countries by their happiness levels, was released at the United Nations at an event celebrating International Day of Happiness on March 20th. The report continues to gain global recognition as governments, organizations and civil society increasingly use happiness indicators to inform their policy-making decisions. Leading experts across fields – economics, psychology, survey analysis, national statistics, health, public policy and more – describe how measurements of well-being can be used effectively to assess the progress of nations. The reports review the state of happiness in the world today and show how the new science of happiness explains personal and national variations in happiness.
# 
# 
# ## Dataset: 
# World Happiness Dataset of 2017
# 
# ## Solution: 
# ask to sir

# ### What you may learn here?
# This notebook is a detailed investigation on world happiness. The art behind happiness has been revealed through amazing visualization. The picture of world happiness, impact factor for increased happiness score and region based happiness investigation will show a interesting scenerio of this dataset. Two regression model constructed at the end for happiness prediction and those works well.
# 

# ## Step 1: Import all the required libraries 
# * __Pandas__ : In computer programming, pandas is a software library written for the Python programming language for data manipulation and analysis and storing in a proper way. In particular, it offers data structures and operations for manipulating numerical tables and time series
# * __Sklearn__ : Scikit-learn (formerly scikits.learn) is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy. The library is built upon the SciPy (Scientific Python) that must be installed before you can use scikit-learn.
# * __Pickle__ : Python pickle module is used for serializing and de-serializing a Python object structure. Pickling is a way to convert a python object (list, dict, etc.) into a character stream. The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.
# * __Seaborn__ : Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

# In[1]:


#Loading libraries 
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn.linear_model 
import pickle
import warnings
warnings.filterwarnings('ignore')


# ## Step 2 : Read all the required data
# In this stage we are going to read our problem dataset and have a look on it.

# In[2]:


#loading training data
try:
    data = pd.read_csv('Data/world-happiness_2017_trainig_dataset.csv')
    print('Data Reading done successfully...')
except (FileNotFoundError, IOError):
    print("Wrong file or file path")   


# In[3]:


(data.head())


# After we read the data, we can look at the data using:
# 

# In[4]:


#count the total number of rows and columns.
print ('The train data has {0} rows and {1} columns'.format(data.shape[0],data.shape[1]))


# ## Step 3: Data Exploration
# 

# Let's proceed and check the distribution of the target variable.

# In[5]:


#Happiness Score
sns.distplot(data['Happiness.Score'])


# We see that the target variable Happiness.Score has a no skewed distribution. This is not always our case for target varibale sometime it happens we have right or left skewed distribution. In that case we'll need to log transform this variable so that it becomes normally distributed. A normally distributed (or close to normal) target variable helps in better modeling the relationship between target and independent variables. In addition, linear algorithms assume constant variance in the error term. Alternatively, we can also confirm this skewed behavior using the skewness metric.

# ## Step 4: Data pre-processing
# 

# In this stage, we'll deal with outlier values, encode variables, impute missing values, and take every possible initiative which can remove inconsistencies from the data set.

# In[6]:


# Encoding Categorical Data
le = preprocessing.LabelEncoder() 
  
data['Country']= le.fit_transform(data['Country']) 

X = data.iloc[:, :-1]
Y = data.iloc[:, 10]


# In[7]:


# check the encoding categorical Data
print(data.head())


# ## Step 5: Split the data into train and test set
# - Now our data is ready to feed to the program. But here we'll split the data into train and test dataset so that after training the model we can test the model on the test dataset and find out how accurate are its predictions.
# - Here we are splitting the data so that the training dataset contains 80% of the data and the test dataset contains 20% of the total data.
# - Here we are using the train_test_split method from the sklearn library. We'll train our model on x_train and y_train, test it on x_test and y_test.
# 
# - test_size: Here we specify the size we want for our test dataset.
# - random_state: When we use a random number generator for number or sequence generation, we give a starting number (AKA seed). When we provide the same seed, every time it’ll generate the same sequence as the first one. That’s why to keep the same random values every time, we give seed as random_state in train_test_split().

# In[8]:


#split the dataset into x and y
x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        X, Y, test_size=0.25,random_state=42)


# We can't see the values of an entire matrix, but by looking at its shape, we decide if we going in the right direction or not. By using ".shape" we can see shape of a matrix and it will also helpful in debugging.

# In[9]:


#display the shapes of the x and y train and test dataset.
print(x_data_train.shape, x_data_test.shape, y_data_train.shape, y_data_test.shape)


# ## Step 6: Train the Model using SVM

# In[10]:


#Using sklearn linear regression model
model = sklearn.linear_model.LinearRegression()
model.fit(x_data_train, y_data_train)
print("Model trained successfully")


# In[11]:


#Score of the Linear Regression
print("Linear Regression:", model.score(x_data_test, y_data_test))
print(model)


# ## Step 7: Save the model in a pickle file
# 
# As Linear regression gave us the highest accuracy we'll go with it and save it to pickle file.
# We save our model to pickle files so that when we want to perform predictions on unseen data, we don't have to train our model again. Any object in python can be pickled so that it can be saved on disk. What pickle does is that it “serializes” the object first before writing it to file. Pickling is a way to convert a python object (list, dict, etc.) into a character stream. 

# In[12]:


# save the model to disk
with open('Model/world_happiness_prediction.pkl','wb') as f:
    pickle.dump(model,f)


# ## Summary 
# or 
# ## Investigation result

# 1. Happiness Ranking basically depends on Economy. Countries having strong GDP rate are happiest countries indeed. 
# 2. Happiness score depends on family, health, freedom, trust on govt. etc. All of these terms have a great linear relationship with happiness score.
# 3. World happiness doesn't define generosity or genesrosity has no remarkable impact on Happiness score.
# 4. Regression Model works quite good in our happiness Dataset. Erros is too small to consider. We can easily check differences on above plots. 
