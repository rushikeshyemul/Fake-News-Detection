import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pickled models and vectorizer
with open('LR.pkl', 'rb') as f:
    LR = pickle.load(f)

with open('DT.pkl', 'rb') as f:
    DT = pickle.load(f)

with open('GBC.pkl', 'rb') as f:
    GBC = pickle.load(f)

with open('RFC.pkl', 'rb') as f:
    RFC = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorization = pickle.load(f)

# Function to preprocess text (wordopt function)
def wordopt(text):
    import re
    import string
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Function to output the label
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "True News"
    else:
        return "Unknown"

# Function to predict news category
def manual_testing(news):
    # Ensure the input is a string and not empty
    if not isinstance(news, str) or not news.strip():
        raise ValueError("Input must be a non-empty string.")

    # Create DataFrame for the input news
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)

    # Preprocess the text
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]

    # Vectorize the input
    new_xv_test = vectorization.transform(new_x_test)

    # Make predictions using different models
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    # Prepare results in a structured format
    results = {
        "Logistic Regression": output_label(pred_LR[0]),
        "Decision Tree": output_label(pred_DT[0]),
        "Gradient Boosting": output_label(pred_GBC[0]),
        "Random Forest": output_label(pred_RFC[0])
    }

    return results

# Custom CSS to add background image and round the text input box with white background
page_bg_img = '''
<style>
/* Background image */
.stApp {
    background-image: url('https://static.vecteezy.com/system/resources/previews/013/831/318/non_2x/breaking-news-template-dark-red-background-vector.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    height: 100vh; /* Ensure full screen coverage */
}

/* Add a white background with rounded corners for the input and predictions section */
.stTextInput > div > textarea {
    background-color: white;
    border-radius: 15px;  /* Round the corners */
    border: 1px solid #ccc;  /* Light border for contrast */
    padding: 10px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    font-size: 1rem;
}

/* Add a white background with rounded corners for the button */
div.stButton > button {
    color: white;
    background-color: #ff4b4b;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    transition: transform 0.3s ease, background-color 0.3s ease;
}
div.stButton > button:hover {
        background-color: white;
       
}

/* Predictions section with a white background and rounded corners */
.stWrite {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
}

/* Title and Subheader Styling */
.stTitle {
    font-family: 'Helvetica Neue', sans-serif;
    font-size: 3rem;
    color: white;
    text-align: center;
    text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.8);
    padding: 10px;
}

/* Centering the content */
.stApp > div:first-child {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}

/* Gradient background for text input */
.stTextInput > div > input {
    background: white;
    color: white;
    border-radius: 10px;
    padding: 10px;
}

</style>
'''

# Inject custom CSS
st.markdown(page_bg_img, unsafe_allow_html=True)

# Streamlit app layout
st.title("Fake News Detection")
st.subheader("Enter a news article to check if it's fake or true:")

# Text input box for user to enter news
user_input = st.text_area("News Article")

# Button to make predictions
if st.button("Check News"):
    try:
        predictions = manual_testing(user_input)

        # Display the predictions
        st.write("### Predictions:")
        for model, result in predictions.items():
            st.write(f"{model}: {result}")
    except ValueError as e:
        st.error(str(e))
