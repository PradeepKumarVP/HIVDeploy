from flask import Flask,request,render_template
import pickle
from sklearn.utils import shuffle

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import pandas as pd

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random

from summarizer import Summarizer,TransformerSummarizer

#Creating GUI with tkinter
import tkinter
from tkinter import *

df = pd.read_excel("Intents.xlsx")
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")


def clean_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_intent(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.20
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    return_list.append({"intent": 'noanswer', "probability": 1})
    for r in results:
        return_list = []
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    if p.min()==0 and p.max()==0:
        return_list = []
        return_list.append({"intent": 'noanswer', "probability": 1})
    return return_list

def getResponse(ints, df):
    tag = ints[0]['intent']
    df = shuffle(df).reset_index(drop=True)
    for i in range(len(df)):
        if(df['Intent'][i]== tag):
            result = df['Answer'][i]
            break
    return result

def chatbot_response(msg):
    ints = predict_intent(msg, model)
    print("ints",ints)
    res = getResponse(ints, df)
    if len(res.split()) > 60:
        res = ''.join(GPT2_model(res, min_length=60))
    return res


app = Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/form_chatbot',methods=['POST','GET'])
def chat():
    
    
    def send():
        msg = EntryBox.get("1.0",'end-1c').strip()
        EntryBox.delete("0.0",END)

        if msg != '':
            ChatLog.config(state=NORMAL)
            ChatLog.insert(END, "You: " + msg + '\n\n')
            ChatLog.config(foreground="#442265", font=("Arial", 10 ))
        
            res = chatbot_response(msg)
            ChatLog.insert(END, "HIV-Bot: " + res + '\n\n')
                
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)
            
    base = Tk()
    base.title("Welcome to HIV-Bot")
    base.geometry("400x500")
    base.resizable(width=FALSE, height=FALSE)

    #Create Chat window
    ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

    ChatLog.config(state=DISABLED)

    #Bind scrollbar to Chat window
    scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
    ChatLog['yscrollcommand'] = scrollbar.set

    #Create Button to send message
    SendButton = Button(base, font=("Arial",10,'bold'), text="Send", width="12", height=5,
                        bd=0, bg="#FF0000", activebackground="#3c9d9b",fg='#ffffff',
                        command= send )

    #Create the box to enter message
    EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
    #EntryBox.bind("<Return>", send)


    #Place all components on the screen
    scrollbar.place(x=376,y=6, height=386)
    ChatLog.place(x=6,y=6, height=386, width=370)
    EntryBox.place(x=128, y=401, height=90, width=265)
    SendButton.place(x=6, y=401, height=90)

    base.mainloop()
    
    return render_template("index.html")

if __name__ == '__main__':
    app.run()
