# python file for chatbot-modualr file
# coding: utf-8

# ## This is modular version of the chatbot FiLe
# As we use  our code for prediction in live environemnt we do reusability of code, so when we import modules from any file it automatically executes the entire file to check that there is no error in what we're importing. Hence it's important to take everything in a module/function and it's executed only when it's called.
#
# ## Dataset :
# The data set contains question and it's corresponding answers. Here we are going to build a chatbot who will answer all the queries of the user based on the data it is trained on.
#
# ## Step 1: Import libraries and files
#
# * __nltk__ : It is a NLP libraries which contains packages to make machines understand human language and reply to it with an appropriate response.
# * __Pandas__ : For analysis and manipulatin of data
# * __Numpy__ : To handel matrices and arrays of large size
# * __Sklearn__ : To easily handel machine learning operations
# * __Pickle__ : To save our model

# In[1]:


import pandas as pd
import numpy as np
import pickle
import sklearn
import nltk
import warnings  # To ignore warnings
warnings.filterwarnings('ignore')
import re


# ## Step 2: Import Data

# In[2]:


# try:
#     FAQs = pd.read_csv('chatbot_faq.csv')
#     greet = pd.read_csv('Greetings.csv')
# except (FileNotFoundError, IOError):
#     print("Wrong file or file path")


# In[3]:


# data = pd.concat([FAQs, greet], ignore_index=True)


# In[4]:


# print("First 5 observations from the data : ")
# print(data.head())


# ## Step 3: Data Cleaning
#
# In data cleaning we remove the unwanted contents from data so that we can get better accuracy.
# Here we are converting each word in its root form. Some times we want our program to recognize that the words “register” and “registation” are just different tenses of the same verb, like for example, we have two questions "How do I register for BridgeLabz?" and "What is the registration processing for BridgeLabz?". Here we want our program to know that registration and register are same so their answers are also same, and that's the reason why we do lemmatizing.
#
# * First we'll tokenzie each word from the dataset.
# - __Tokenizing__ : This breaks up the strings into a list of words or pieces based on a specified pattern using Regular Expressions aka RegEx.
# - eg : white brown fox = ‘white’, ‘brown’,’fox’
#
# * After we tokenize, we will start cleaning up the tokens by Lemmatizing.
# - __Lemmatizing__ : Lemmatizing is the process of converting a word into its root form.
# - e.g., "Playing", "Played" = "play".
#
# In cleanup() function, we are first tokenizing the sentence (seperating each word in sentence) and then steeming (converting a word into its root form) and at the end combine all the words to form a sentence.

# In[5]:


import re
class Data_Cleanig:
    def data_cleanup(self, sentence):
        TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
        cleaned_text = re.sub(TEXT_CLEANING_RE, ' ', str(sentence).lower()).strip()
        word_tok = nltk.word_tokenize(cleaned_text)
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(w) for w in word_tok]
        return ' '.join(lemmatized_words)

# def cleanup( sentence):
#     word_tok = nltk.word_tokenize(sentence)
#     stemmer = nltk.stem.lancaster.LancasterStemmer()
#     stemmed_words = [stemmer.stem(w) for w in word_tok]
#     return ' '.join(stemmed_words)
# cleaning = Data_Cleanig()


#  Pass each question to the cleaning funtion defined above

# In[6]:


# questions_cleaned = []
# questions = data['Question'].values
# questions
# for question in questions:
#     questions_cleaned.append(cleaning.data_cleanup(question))


# In[7]:


# print("Following are the questions before and after cleaning the data: \n")
# for i in range(4):
#     print(data.Question.iloc[i], "===>", questions_cleaned[i],"\n")


# ## Step 4 : Data preprocessing
# - After removing unwanted data let's do some steps to make our data understandable for our program. That's why we do preprocessing.
# - Here we are dealing with text data, we can understant it but our machines can't. So we need to convert the data from text to numeric form.
# - Vectorization :The process of converting NLP text into numbers is called vectorization in ML.
# - TF-IDF : TF-IDF stands for term frequency-inverse document frequency. It tell how important a word is in a sentence. The importance of a word depends on the number of times it occured in a sentence. To understand it, let's see each term:
# - __Term Frequency(TF)__ : How frequently a word appears in a sentence. We can measure it by an equation,
#
# - TF = __(Total number of times the word "W" occured in the sentence) / (Total number of words in the sentence)__
# - __Inverse Document Frequency (IDF)__ : How common is a word across all the sentences.
# - IDF = __log( (Total number of sentences) / (Number of sentences with word "W" in it))__
# * Apply vecorization on the cleaned questions
# * Here we have used tfidf vectorizer
# * It’ll see the unique words in the complete para or content given to it and then does one hot encoding accordingly. Also it removes the stopwords and stores the important words which might be used less but gives us more better features. And stores the frequency of the words.

# In[8]:


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

#
# preprocessing = Preprocessing()


# In[9]:


# _features, vectorizer = preprocessing.vectorize(questions_cleaned)


# In[10]:


def remove_stopwords(data):
    import re
    stop_words = set(nltk.corpus.stopwords.words('english'))
    lemma_function = nltk.stem.WordNetLemmatizer()
    sentences = []
    for text in data:
        lemma = []
        for token in text.split(' '):
            if token not in stop_words:
                lemma.append(lemma_function.lemmatize(token))
        sentences.append(" ".join(lemma))
    return sentences


# In[11]:


# print("Original Data : ", data.Question[:1].values)
# print('\nCleaned data : ', questions_cleaned[0])
# d = remove_stopwords([questions_cleaned[0]])
# print('\nAfter removing stopwords : ', d)
# print("\nVectorized Data : \n", _features[0])


# ## Step 5: Split data into train and test datasets
# - Now our data is ready to feed to the program. But here we'll split the data into train and test dataset so that after training the model we can test the model on test dataset and find out how accurate are its predictions.
# * Here we are spliting the data so that train dataset contains 75% of the data and test dataset contains 25% of the total data.

# In[12]:


# # Split the dataset into train and test dataset
# _labels = data.Class.values
# feature_train, feature_test, label_train, label_test = sklearn.model_selection.train_test_split(_features, _labels, test_size=.25, random_state=42)


# ## Step 6: Train the model
# - Now we'll apply an machine learning algorithm to our processed data.
# - Here we've used SVC algorithm to train our model with linear kernel and fit the data.
# - SVC is a classification algorithm which will classify the category of the question.
# - Score : Returns the mean accuracy on the given test data and labels.

# In[13]:


# model = sklearn.svm.SVC(kernel='linear')
# model.fit(feature_train, label_train)
# print("SVC:", model.score(feature_test, label_test))
# prediction = model.predict(feature_test)


# ## Step 7: Save model to the pickle file
#
# - Here we are goining to save the model we trained into a pickle file, so that we can use it without training on new unseen data
# - So we just load the pickle file, extract the model and apply the data on it.

# In[14]:


# with open('model.pkl','wb') as f:
#     pickle.dump(cleaning, f)
#     pickle.dump(preprocessing, f)
#     pickle.dump(model,f)


# In[15]:


def check_answer(cleaning,preprocessing,vectorizer,model,data,usr):
    cleaned_usr = cleaning.data_cleanup([usr])
    t_usr_array = preprocessing.query(cleaned_usr, usr, vectorizer)
    prediction = model.predict(t_usr_array)[0]
    questionset = data[data['Class']==prediction]


    cos_sims = []
    for question in questionset['Question']:
        cleaned_question = cleaning.data_cleanup(question)
        question_arr = preprocessing.query(cleaned_question,question, vectorizer)
        sims = sklearn.metrics.pairwise.cosine_similarity(question_arr, t_usr_array)
        cos_sims.append(sims)

    if len(cos_sims) > 0:
        ind = cos_sims.index(max(cos_sims))
        # print(data['Answer'][questionset.index[ind]])
        return data['Answer'][questionset.index[ind]]


# ## Step 8: Test the model
#
# Here we have trained and tested the model

# In[16]:


def execute():
    try:
        FAQs = pd.read_csv('Data/chatbot_faq.csv')
        greet = pd.read_csv('Data/Greetings.csv')
    except (FileNotFoundError, IOError):
        print("Wrong file or file path")

    data = pd.concat([FAQs, greet], ignore_index=True)

    print("First 5 observations from the data : ")
    print(data.head())

    cleaning = Data_Cleanig()

    questions_cleaned = []
    questions = data['Question'].values
    # questions
    for question in questions:
        questions_cleaned.append(cleaning.data_cleanup(question))

    preprocessing = Preprocessing()

    _features, vectorizer = preprocessing.vectorize(questions_cleaned)

    _labels = data.Class.values
    feature_train, feature_test, label_train, label_test = sklearn.model_selection.train_test_split(_features, _labels,
                                                                                                    test_size=.25,
                                                                                                    random_state=42)
    model = sklearn.svm.SVC(kernel='linear')
    model.fit(feature_train, label_train)
    print("SVC:", model.score(feature_test, label_test))
    prediction = model.predict(feature_test)

    with open('model.pkl', 'wb') as f:
        pickle.dump(cleaning, f)
        pickle.dump(preprocessing, f)
        pickle.dump(vectorizer,f)
        pickle.dump(model, f)

    check_answer(cleaning,preprocessing,vectorizer,model,data,'What is fellowship program ')

