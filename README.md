# Chatbot for Emotional Support
[Streamlit App](https://chatbot-oaf4caatfj45wam48qqqtj.streamlit.app/)

## Overview
This project implements an interactive chatbot designed to provide emotional support and respond empathetically to user input. The chatbot leverages Natural Language Processing (NLP) techniques to understand user intents and generate appropriate responses. Built with the `nltk` library for NLP, `scikit-learn` for machine learning, and `streamlit` for the web interface, it aims to assist users by listening, understanding, and offering thoughtful replies.

---

## Features
- Understands various user intents related to emotions, motivation, and support (e.g., "greeting", "fear", "confidence").
- Provides responses that help users manage emotions and find comfort.
- Maintains a conversation history, allowing users to review previous interactions.
- Built using Python with popular libraries for NLP and machine learning.

---

## Technologies Used
- **Python**
- **NLTK**: For natural language processing and tokenization.
- **Scikit-learn**: For training the machine learning model (Logistic Regression).
- **Streamlit**: For creating the web interface.
- **JSON**: For structuring and loading the intents data.

---

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
```python
import nltk
nltk.download('punkt')
```

---

## Usage
To launch the chatbot application, run the following command:
```bash
streamlit run app.py
```

After starting the application, access it through the web interface. Enter your messages in the input box and hit Enter to receive responses from the chatbot.

---

## Intents Data
The chatbot's behavior is guided by the `intents.json` file. This file includes various tags, patterns, and responses that define how the chatbot responds to user inputs. You can modify this file to add new intents or adjust existing ones to fine-tune the chatbot's responses.

---

## Contributing
Contributions are welcome! If you have ideas for new features, enhancements, or improvements, feel free to open an issue or submit a pull request.

---

## Acknowledgments
- **NLTK** for its powerful natural language processing tools.
- **Scikit-learn** for its machine learning algorithms used for intent classification.
- **Streamlit** for creating the user-friendly web interface.

---

*Replace `<repository-url>` and `<repository-directory>` with the actual URL of your repository and the name of the project directory. Customize any sections to better align with your specific project details.*
