# 🧠 Deep Learning Mental Health Chatbot (V2)

A Neural Network-powered chatbot built with **TensorFlow** and **NLTK** to provide mental health support.
**Live Demo:** [Click Here to Chat](https://student-chatbot-deeplearning.streamlit.app/)

## 🚀 Project Overview
Unlike simple rule-based bots, this project uses **Deep Learning** to understand the *intent* behind a user's message. It is trained on a custom dataset of mental health conversations to recognize patterns like sadness, stress, anxiety, and greetings.

### 🤖 How it Works (The Architecture)
The system uses a **Sequential Neural Network** built with Keras/TensorFlow:
1.  **Input Layer:** Receives the text data converted into a "Bag of Words".
2.  **Hidden Layers:** Two layers of neurons (128 & 64 units) with **Dropout** to prevent overfitting.
3.  **Output Layer:** Uses a **Softmax** activation function to predict the probability of the user's intent (e.g., 85% Sadness, 15% Stress).

### 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **NLP:** NLTK (Tokenization, Lemmatization, Bag-of-Words)
* **Language:** Python 3.12
* **Frontend:** Streamlit
* **Deployment:** Streamlit Cloud

## 📂 Project Structure
* `chatbot_model.h5`: The trained Neural Network ("The Brain").
* `intents.json`: The training dataset containing patterns and responses.
* `words.pkl` & `classes.pkl`: Preprocessed vocabulary files required for the model to understand new text.
* `app.py`: The main application code connecting the Brain to the Web Interface.

## 📊 Model Performance
* **Accuracy:** ~92% on the training dataset.
* **Epochs:** Trained for 200 epochs using Stochastic Gradient Descent (SGD).
## ✍️ Author
**Upamada Ekanayake** - *AI & Data Science Student*
