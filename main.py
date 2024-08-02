import os
import streamlit as st
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import pandas as pd

# Load environment variables
load_dotenv()

# Set up Gemini API key
gemini_api_key = os.getenv('GEMINI_API_KEY')
if gemini_api_key is None:
    st.error("GEMINI_API_KEY is not set")
else:
    st.success("GEMINI_API_KEY is set")

# Database configuration
config = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME')
}

def get_tables():
    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES;")
        tables = [table[0] for table in cursor.fetchall()]
        return tables
    except Error as e:
        st.error(f"Error: {e}")
        return []
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_metadata(table_name):
    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        query = f"SHOW FULL COLUMNS FROM `{table_name}`"
        cursor.execute(query)
        columns = cursor.fetchall()
        metadata = []
        for column in columns:
            metadata.append(f"Field: {column[0]}, Type: {column[1]}, Collation: {column[2]}, "
                            f"Null: {column[3]}, Key: {column[4]}, Default: {column[5]}, "
                            f"Extra: {column[6]}, Privileges: {column[7]}, Comment: {column[8]}")
        return metadata
    except Error as e:
        st.error(f"Error: {e}")
        return []
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def generate_prompt(user_prompt):
    prompt = (
        f"You need to analyze this and write an SQL query in normal text (not even markdown) to answer the below natural language question. "
        f"Don't give any explanation, just write the query.\n"
        f"Question: {user_prompt}"
    )
    return prompt

def execute_sql_query(query):
    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        df = pd.DataFrame(results, columns=columns)
        return df
    except Error as e:
        st.error(f"Error executing query: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

@st.cache_resource
def load_data():
    tables = get_tables()
    table_metadata = {}
    for table in tables:
        table_metadata[table] = get_metadata(table)
    
    documents = []
    for table_name, fields in table_metadata.items():
        table_info = f"Table: {table_name}\n"
        table_info += "\n".join(fields)
        documents.append(Document(text=table_info))

    # Set up Gemini LLM
    llm = Gemini(
        model="models/gemini-1.5-flash",
        api_key=gemini_api_key
    )

    # Set up Gemini Embedding model
    embed_model = GeminiEmbedding(
        model_name="models/embedding-001",
        api_key=gemini_api_key)

    # Configure settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    
    index = VectorStoreIndex.from_documents(documents)
    return index

st.title("Database Metadata Chat")

index = load_data()

query_engine = index.as_query_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know about the database?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        updated_prompt = generate_prompt(prompt)
        response = query_engine.query(updated_prompt)
        generated_sql = response.response.strip()
        st.markdown(f"Generated SQL Query:")
        st.code(generated_sql, language="sql")
        
        st.markdown("Query Results:")
        results = execute_sql_query(generated_sql)
        if results is not None and not results.empty:
            st.dataframe(results)
        elif results is not None and results.empty:
            st.info("The query returned no results.")
        else:
            st.error("Failed to execute the query. Please check the SQL syntax or database connection.")
    
    st.session_state.messages.append({"role": "assistant", "content": f"SQL Query: {generated_sql}\n\nResults: {results.to_string() if results is not None else 'Query execution failed'}"})

st.sidebar.header("Sample Queries")

sample_queries = [
    "Sales people working in tech companies"
]

for i, query in enumerate(sample_queries):
    st.sidebar.subheader(f"Sample Query {i+1}")
    st.sidebar.text(query)
    