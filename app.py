import streamlit as st
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

# 1. Download NLTK data (needed for the server)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# 2. Load the Brain & Tools
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the rules (Responses)
intents = json.loads(open('intents.json').read())

# --- HELPER FUNCTIONS (The Translator) ---

def clean_up_sentence(sentence):
    # Split sentence into words
    sentence_words = nltk.word_tokenize(sentence)
    # Shorten words (running -> run)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    # Create a bag of 0s
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    
    # Set 1 if word is found
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                
    return np.array(bag)

def predict_class(sentence):
    # 1. Translate sentence to numbers
    bow = bag_of_words(sentence)
    
    # 2. Ask the brain to predict
    res = model.predict(np.array([bow]))[0]
    
    # 3. Get the answer (Tag)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Sort by probability (most likely first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    # Check our JSON for the response matching the tag
    for i in list_of_intents:
        if i['tag'] == tag:
            result = np.random.choice(i['responses'])
            break
    return result

# --- WEBSITE UI ---

st.title("🤖 Deep Learning Mental Health Chatbot")
st.write("I am a Neural Network. Talk to me!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is on your mind?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI Logic
    try:
        # 1. Predict
        ints = predict_class(prompt)
        # 2. Get Response
        res = get_response(ints, intents)
    except:
        res = "I am not sure I understand. Can you rephrase?"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(res)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": res})
