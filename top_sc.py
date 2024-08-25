import streamlit as st
import pandas as pd
import numpy as np
import requests
# from sentence_transformers import SentenceTransformer
import io
import matplotlib.pyplot as plt
import re
import chromadb
from chromadb.config import Settings

def extract_code_block(text):
    pattern = r'```(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None

def add_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffecd1 ;
        }
        header{
            border-bottom:1 px solid;
            border-bottom-color:white !important;
            background-color: #c27f7c !important;
        }
        .header{
            background-color: #c27f7c !important;
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
            background-color:#c27f7c;
            padding: 10px;
            border-radius: 5px;
        }
        .stMarkdown p {
            color: #fff;
        }
        .stChatMessage {
            background-color: #c27f7c;
            color: #fff;
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
            color:# c27f7c ;
        }
        .header img {
            height: 50px;
        }
        .stChatFloatingInputContainer{
            background-color: #ffecd1 !important;
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

# Function to get inference response
def get_inference_response(prompt):
    api_token = "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"
    api_url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    data = {
        "inputs":  prompt,
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.9999,
            "frequence_penalty":1,
            "max_new_tokens": 300
        }   
    }
    response = requests.post(api_url, headers=headers, json=data)
    print(response.json())
    result = response.json()[0]["generated_text"].replace(data['inputs'], " ")
    return result

# Load the CSV file with dtype specification and handling mixed types
@st.cache_data
def load_data():
    df = pd.read_csv('20240521_082430.csv', dtype={'City': str, 'Sector': str, 'question': str, 'Denominator': str, 'UOM': str, 'Result': str, 'encoding': str})
    # Convert encoding column to list of floats
    # df['encoding'] = df['encoding'].apply(lambda s: [float(x) for x in s.strip().replace('[', '').replace(']', '').split(',')])
    return df

# model = SentenceTransformer('all-MiniLM-L6-v2')
# Initialize ChromaDB client and collection
client = chromadb.PersistentClient(path="my_vectordb")
collection_name = "city_macroeconomics"
collection = client.get_or_create_collection(collection_name)
from tqdm import tqdm
# Check if collection is empty; if so, populate it with data from the DataFrame
if collection.count() == 0:
    print("Adding to collection")
    df = load_data()
    print("Loaded Data")
    print("Adding to Chroma....")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        print(index)
        collection.add(
            documents=[str(row[["City", "Sector", "question", "Result"]])],
            ids=[str(index )]
        )
    print("done")

def find_top_similar_rows(input_text, top_n=233):
    # input_embedding = model.encode(input_text, convert_to_tensor=True)
    results = collection.query(
        query_texts=[input_text],
        n_results=top_n
    )
    top_5_strings = [
    f"{i + 1}. " + ', '.join(f"{key}: {value.split(', ')[0]}" if 'dtype' in value else f"{key}: {value}" for key, value in [line.split(None, 1) for line in res.split('\n') if line.strip()] if key != 'Name:')
    for i, res in enumerate(results['documents'][0])
    ]
    
    top_responses = '\n'.join(top_5_strings)
    prompt = f"""
<|start_header_id|>system<|end_header_id|>

dataset to be used:
{top_responses}
Tou are an assistant helping users of a website into presenting the most relevant information in the best form.
The User is being shown the output of your code in a completely independent environment, hence POSITIVELY hardcode ALL relevant numbers and values for the graphs. 
You can also be asked to make subjective reports about certain metrics. Take a look at the dataset and come to a logical conclusion.
If the query is in a non-English language, reply in that language.
The Format of the dataset is 'City','Context','Result'.
Do not include outside information in your response, whatever information you query to present to the user must be from the dataset.
For state-specific questions, infer all the cities given from the dataset and provide the relevant information. 
Wrap python code with backticks. respond with only the code when asked to make a graph.

Example of a Wrong code response asking to scatter plot the populations in haryana:
```python
import matplotlib.pyplot as plt

haryana_cities = [city for city in dataset if city[0].startswith('H')]
populations = [int(city[2]) for city in dataset if city[0].startswith('H')]

plt.figure(figsize=(10, 5))
plt.scatter(haryana_cities, populations, color='green')
plt.xlabel('Cities')
plt.ylabel('Population')
plt.title('Population of Cities in Haryana')
plt.show()
```
Example of a Right code response:
```python
import matplotlib.pyplot as plt
haryana_cities = ['Rohtak', 'Gurugram','Faridabad','Sonipat','Hisar'] # add more as per the values given in the dataset
populations = [452596,1540660,1414050,330285,301383] # add the respective populations
plt.figure(figsize=(10, 5))
plt.scatter(haryana_cities, populations, color='green')
plt.xlabel('Cities')
plt.ylabel('Population')
plt.title('Population of Cities in Haryana')
plt.show()
```
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is the population of the cities of California?
- New York: 8.3 million
- San Francisco: 0.88 million
- Los Angeles: 3.9 million
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
The populations of relevant cities in California are:
- San Francisco: 0.88 million
- Los Angeles: 3.9 million
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Can you show a graph of the populations of the cities in California?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

```python
import matplotlib.pyplot as plt
cities = ['San Francisco', 'Los Angeles']
populations = [0.88, 3.9]
plt.figure(figsize=(10, 5))
plt.bar(cities, populations, color=['green', 'red'])
plt.xlabel('Cities')
plt.ylabel('Population (millions)')
plt.title('Population of Cities in California')
plt.show()
```
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Can you show a scatter graph of the populations of the cities in California?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```python
import matplotlib.pyplot as plt
cities = ['San Francisco', 'Los Angeles']
populations = [0.88, 3.9]
plt.figure(figsize=(10, 5))
plt.scatter(cities, populations, color=['green', 'red'])
plt.xlabel('Cities')
plt.ylabel('Population (millions)')
plt.title('Population of Cities in California')
plt.show()
```
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{input_text}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    return get_inference_response(prompt)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi I can help you query about the city macroeconomics. \n Ask me a question like: Total number of MSME clusters in Gurugram"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Enter your Query:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = find_top_similar_rows(prompt)
    try:
        code = extract_code_block(response).replace('Python','').replace('python','')
        print(code)

        try:
            # Prepare a namespace for the exec function
            namespace = {}

            # Capture the generated plot as a PNG image
            fig = None
            exec(code, {"plt": plt}, namespace)

            # Check if a figure was created
            fig = namespace.get('fig', plt.gcf())

            # Convert plot to PNG image
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)

            # Display the image in Streamlit
            st.image(buf)

        except Exception as e:
            st.error(f"Error running code block: {e}")
    except AttributeError:
        print("No Code Found")
    with st.chat_message("assistant"):
        response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

add_custom_css()