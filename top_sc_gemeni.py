import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
import requests
from gensim.models import KeyedVectors
import os
import json

def add_custom_css():
    st.markdown(
        """
       
        <style>
        .stApp {
            
            background-color: #ffecd1 ;
        }
        header{
        color:#000 !important;
        background-color: #ffecd1 !important;
        }
        ::placeholder{
            color:white !important;
        } 
        .stButton>button {
            background-color: #a32020;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #dc6900;
            color: white;
        }
        .stTextArea>div>div>textarea {
            background-color: #e0301e;
            color: white;
        }
        .stMarkdown {
            background-color: #c27f7c;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        .stMarkdown p {
            color: white;
        }
        .stChatMessage {
            background-color: #c27f7c;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        .stChatInput{
            background-color:#4f3231 !important;
        }
        
         div[data-testid="stBottom"] > div {
        background-color: #ffecd1 !important;
        color: #000000 !important;
    }
    .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border-bottom: 2px solid #a32020;
        }
        .header h1 {
            margin: 0;
            color: white;
        }
        .header img {
            height: 50px;
        }
        </style>
        """,
        
        unsafe_allow_html=True
    )
# Add header with logo and title
st.markdown(
    """
    <div class="header">
        <h1>Amplifi Chatbot</h1>
        <img src="https://1000logos.net/wp-content/uploads/2021/05/PwC-logo.png" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)
# Load API key
GEMENI_API_KEY = os.getenv('GEMENI')
if not GEMENI_API_KEY:
    print("GEMENI API_KEY not set. Please set the GEMENI environment variable with the API key.")
    st.stop()
else:
    print(f"API KEY: {GEMENI_API_KEY} found")

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load the pre-trained word2vec model
@st.cache_resource
def load_word2vec_model():
    print("Loading Word2Vec Model...")
    model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
    return model

word2vec_model = load_word2vec_model()

# Load the CSV file
@st.cache_data
def load_data():
    df = pd.read_csv('20240521_082430.csv')
    return df

df = load_data()
df['combined_text'] = df.astype(str).agg(' '.join, axis=1)

# Function to lemmatize text using spaCy
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

# Function to get inference response
def get_inference_response(prompt):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={GEMENI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        response_json = response.json()
        return response_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No content received')
    else:
        return f"Error: {response.status_code}"

# Function to compute the average word2vec vector for a text
def compute_average_word2vec(text, model, num_features):
    words = text.split()
    feature_vec = np.zeros((num_features,), dtype="float32")
    n_words = 0
    for word in words:
        if word in model.key_to_index:
            n_words += 1
            feature_vec = np.add(feature_vec, model.get_vector(word))
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

n = df['City'].nunique()
# Function to find the top similar rows
def find_top_similar_rows(input_text, df, model, top_n=n, return_n=n, chunk_size=1000):
    num_features = model.vector_size
    combined_texts = df['combined_text'].tolist()

    if input_text:
        lemmatized_input_text = lemmatize_text(input_text)
        combined_texts.append(lemmatized_input_text)
        
        vectors = np.array([compute_average_word2vec(text, model, num_features) for text in combined_texts])
        
        input_vector = vectors[-1]
        similarities = []

        # Calculate similarities in chunks
        for i in range(0, len(vectors) - 1, chunk_size):
            end_idx = min(i + chunk_size, len(vectors) - 1)
            chunk_vectors = vectors[i:end_idx]
            chunk_similarities = cosine_similarity([input_vector], chunk_vectors)
            similarities.extend(chunk_similarities[0])
        
        similarities = np.array(similarities)
        top_indices = similarities.argsort()[-top_n:][::-1]
        top_similar_rows = df.iloc[top_indices]
        
        top_5_rows = top_similar_rows.head(return_n)
        top_5_strings = [f"{i + 1}. {' '.join([f'{row[col]},' for col in ['City', 'question', 'Result'] if col != 'combined_text'])}" for i, (index, row) in enumerate(top_5_rows.iterrows())]
        
        top_responses = '\n'.join(top_5_strings)
        prompt = f"""
        You are a bot helping users answer questions about a dataset.
        There is a list of Cities in the dataset. 
        The Dataset follows the Structure 'City','Context','Result'. The Context essentially tells us What is in the Result.
        
        The user's query is "{input_text}" answer the question asked by the user using this dataset:
        ----
          {top_responses}
        ----.
        Be very specific about your result, Be sure to mention the context as well with your result.
        Find the most accurate answer by looking at the most relevant context.
        For eg. If the user is asking for highest energy consumption, look at all the cities with that same context and return the city with the highest result.
        """
        print(prompt)
        
        return get_inference_response(prompt)
    else:
        return ""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi I can help you query about the city macroeconmics. \n Ask me a question like: Total number of MSME clusters in Gurugram"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Enter your Query:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = find_top_similar_rows(prompt, df, word2vec_model)
    
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


add_custom_css()