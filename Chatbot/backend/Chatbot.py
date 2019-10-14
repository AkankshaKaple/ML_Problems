#!/usr/bin/env python
# coding: utf-8

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

# try:
#     FAQs = pd.read_csv('Data/chatbot_faq.csv')
#     greet = pd.read_csv('Data/Greetings.csv')
# except (FileNotFoundError, IOError):
#     print("Wrong file or file path")
#
# data = pd.concat([FAQs, greet], ignore_index=True)
#
# print("First 5 observations from the data : ")
# print(data.head())

import re
class Data_Cleanig:
    def data_cleanup(self, sentence):
        TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
        cleaned_text = re.sub(TEXT_CLEANING_RE, ' ', str(sentence).lower()).strip()
        word_tok = nltk.word_tokenize(cleaned_text)
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(w) for w in word_tok]
        return ' '.join(lemmatized_words)

# cleaning = Data_Cleanig()
#
# questions_cleaned = []
# questions = data['Question'].values
# # questions
# for question in questions:
#     questions_cleaned.append(cleaning.data_cleanup(question))

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

# preprocessing = Preprocessing()
#
# _features, vectorizer = preprocessing.vectorize(questions_cleaned)

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

# _labels = data.Class.values
# feature_train, feature_test, label_train, label_test = sklearn.model_selection.train_test_split(_features, _labels, test_size=.25, random_state=42)
#
# model = sklearn.svm.SVC(kernel='linear')
# model.fit(feature_train, label_train)
# print("SVC:", model.score(feature_test, label_test))
# prediction = model.predict(feature_test)
#
# with open('model.pkl','wb') as f:
#     pickle.dump(cleaning, f)
#     pickle.dump(preprocessing, f)
#     pickle.dump(model,f)

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
# check_answer('What is fellowship program ')

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

