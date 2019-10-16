from Chatbot import *

# execute()

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

question = "what is fellowship?"
answer = check_answer(cleaning, preprocessing, vectorizer, model, data, question)

print(answer)