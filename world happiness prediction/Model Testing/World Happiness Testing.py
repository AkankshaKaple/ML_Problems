
# coding: utf-8

# # Model testing for World Happiness
# - Here we are going apply the model we created on live data.

# ## Step 1: Import libraries
# - __Pickle__: To load model from pickle file.
# * __Pandas__ : In computer programming, pandas is a software library written for the Python programming language for data manipulation and analysis and storing in a proper way. In particular, it offers data structures and operations for manipulating numerical tables and time series

# In[1]:


import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing


# ## Step 2: Import Model
# - Now we'll import the model 

# In[2]:


with open('../Model Training/Model/world_happiness_prediction.pkl','rb') as file:
    model = pickle.load(file)


# ## Step 3: Apply model on the test data

# In[3]:


#loading testing data
try:
    data = pd.read_csv('../Model Testing/Data/world-happiness_2017_testing_dataset.csv')
    print('Data Reading done successfully...')
except (FileNotFoundError, IOError):
    print("Wrong file or file path")   


# In[4]:


#show the first five rows of the data
data.head()


# In[5]:


#split the dataset into x_data_test and y_data_test
x_data_test = data.iloc[:, :-1]
y_data_test = data.iloc[:, 10]


# In[6]:


#function to display fetch data from the specific country name
def specific_country_data(country_name):
    return x_data_test.loc[data['Country'] == country_name]


# In[7]:


#call the function
specific_country_data('India')


# In[8]:


#encode the Country column
le = preprocessing.LabelEncoder() 
x_data_test['Country']= le.fit_transform(x_data_test['Country']) 
x_data_test = x_data_test.to_numpy()


# In[9]:


#predict the world happiness 
model.predict(x_data_test)

