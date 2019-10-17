from flask import Flask, render_template, request
import re
import nltk
import pickle

app = Flask(__name__, template_folder='template', static_folder='style')

"""
    Import model for prediction
"""
path = 'model/logistic_model.pkl'
with open(path, 'rb') as file:
    bow_obj = pickle.load(file)
    model = pickle.load(file)
print("Model Imported")


@app.route("/")
def index():
    return render_template('input_tweet.html')


def tokenization(data):
    """
    :param data: It will receive the tweet and perform tokenization and remove the stopwords
    :return: It will return the cleaned data
    """
    stop_words = set(nltk.corpus.stopwords.words('english'))
    stop_words.remove('no')
    stop_words.remove('not')

    tokenizer = nltk.tokenize.TweetTokenizer()

    document = []
    for text in data:
        collection = []
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            if token not in stop_words:
                if '#' in token:
                    collection.append(token)
                else:
                    collection.append(re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", " ", token))
        document.append(" ".join(collection))
    return document


def lemmatization(data):
    """
    :param data: Receive the tokenized data
    :return: Return the cleaned data
    """
    lemma_function = nltk.stem.wordnet.WordNetLemmatizer()
    sentence = []
    for text in data:
        document = []
        words = text.split(' ')
        for word in words:
            document.append(lemma_function.lemmatize(word))
        sentence.append(" ".join(document))
    return sentence


@app.route("/index", methods=['GET', 'POST'])
def get_tweets():
    """
        Here we'll perform predictions on the data given by the tweeter.
    """
    my_tweet = request.form['my_tweet']
    tokenized_data = tokenization([my_tweet])
    lemmatized_data = lemmatization(tokenized_data)
    temp = bow_obj.transform(lemmatized_data)
    pred = model.predict(temp)
    if pred == 0:
        print('Positive')
        return render_template('input_tweet.html', my_tweet='Positive')
    else:
        print('Negative')
        return render_template('input_tweet.html',my_tweet='Negative')


if __name__ == '__main__':
    app.run(debug=True)


