import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./in.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 15))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Define special cases with specific responses
special_cases = {
      "help": "Of course! What do you need help with?",
    "what can you do": "I can provide support, answer questions, and offer guidance. Just ask me anything!",
    "what is your work": "My work is to listen, understand, and provide helpful responses to your questions and concerns.",
    "help me": "I'm here to help! What do you need assistance with?",
    "do you know abt lahari?": "yeahh...she is so bad",
    }

def chatbot(input_text):
    # Check for special cases first
    lower_input_text = input_text.lower()
    if lower_input_text in special_cases:
        return special_cases[lower_input_text]
    
    # If not a special case, proceed to check intents
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

    return "I'm not sure how to respond to that. Can you try asking something else?"
        
counter = 0

def main():
    global counter
    st.title("Your listening ear")

    # Create a sidebar menu with options
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Hello! I'm here to provide you with emotional support and a listening ear. Feel free to share your thoughts, worries, or anything on your mind. Type a message and press Enter to start our conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        # with st.beta_expander("Click to see Conversation History"):
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on various intents. The chatbot is built using a combination of Natural Language Processing (NLP) techniques and machine learning algorithms, specifically Logistic Regression, to identify and extract intents and entities from user input. The interactive interface is powered by Streamlit, a Python library that facilitates the development of user-friendly web applications. This chatbot is designed to provide emotional support, offering thoughtful responses and assistance to users experiencing different emotions.")

        st.subheader("Project Overview:")

        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
        """)

        st.subheader("Dataset:")

        st.write("""
        The dataset used in this project is designed to train a chatbot to understand and respond to user inputs based on various intents. The data is stored in a structured JSON format, with the following components:

        - **Intents**: Represents the type or category of user input, such as greetings, questions about emotions, or general support (e.g., "greeting", "fear", "confidence").
        - **Patterns**: The different ways users can phrase their inputs that correspond to a particular intent (e.g., "Hello", "I feel scared", "I am confident").
        - **Responses**: Predefined replies that the chatbot uses to respond to user inputs related to each intent (e.g., "Hello! How can I assist you today?", "Take a deep breath and stay calm.", "You are capable and strong. Believe in yourself!").
        - **Special Cases**: Specific user queries or phrases that have unique, predefined responses (e.g., "What can you do?").

        The data format helps the chatbot to learn which responses to provide based on the user's intent, enabling it to offer emotional support and assistance.
        """)

    


        st.subheader("Streamlit Chatbot Interface:")

        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

        st.subheader("Conclusion:")

        st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This chatbot is designed to provide emotional support, offering empathetic and thoughtful responses to users. The project can be extended by adding more data, using more sophisticated NLP techniques, and incorporating deep learning algorithms for enhanced performance.")

if __name__ == '__main__':
    main()
