from flask import Flask,render_template,request
import pickle
import pandas as pd
from Chatbot import Data_Cleanig, Preprocessing, check_answer
import sklearn
import json

from flask_cors import CORS

app = Flask(__name__)
CORS(app)


try:
    FAQs = pd.read_csv('Data/chatbot_faq.csv')
    greet = pd.read_csv('Data/Greetings.csv')
except (FileNotFoundError, IOError):
    print("Wrong file or file path")

data = pd.concat([FAQs, greet], ignore_index=True)

f=open("model.pkl","rb")
cleaning=pickle.load(f)
preprocessing=pickle.load(f)
vectorizer = pickle.load(f)
model=pickle.load(f)


@app.route("/")
def greeting():
    return "<h1 style='color:green'>Welcome to chat with Brigibot</h1>"

@app.route('/chat', methods=['GET'])
def chatpage():
    return render_template('/chat.html')

@app.route('/ask/<question>', methods=['GET','POST'])
def data_download(question):
    # if question != " " or "":
    # question = request.form['question']
    # question = "what is fellowship?"
    print(question)
    answer = check_answer(cleaning, preprocessing, vectorizer, model, data, question)

    # dict = {}
    # dict =
    answer = {'data':{'answer':answer}}
    return json.dumps(answer)
            # render_template("chat.html", answer=answer)
    # else:
    #     question = request.args.get('question')
    #     answer = check_answer(cleaning, preprocessing, vectorizer, model, data, question)
    #     return answer


if __name__ == "__main__":
    app.run(host='127.0.0.1')