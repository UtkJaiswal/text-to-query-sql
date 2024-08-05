import os
import streamlit as st
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import psycopg2
from psycopg2 import Error
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

connection_string = os.getenv('DATABASE_URL')


def get_tables():
    conn = None
    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [table[0] for table in cursor.fetchall()]
        return tables
    except Error as e:
        st.error(f"Error: {e}")
        return []
    finally:
        if conn:
            cursor.close()
            conn.close()

def get_metadata(table_name):
    conn = None
    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        query = f"""
            SELECT 
                column_name, 
                data_type, 
                collation_name,
                is_nullable,
                column_default,
                (SELECT pg_catalog.col_description(c.oid, cols.ordinal_position::int)
                FROM pg_catalog.pg_class c
                WHERE c.oid = (SELECT ('"' || cols.table_name || '"')::regclass::oid)
                    AND c.relname = cols.table_name) AS column_comment
            FROM information_schema.columns cols
            WHERE table_name = '{table_name}'
        """
        cursor.execute(query)
        columns = cursor.fetchall()
        metadata = []
        for column in columns:
            metadata.append(f"Field: {column[0]}, Type: {column[1]}, Collation: {column[2]}, "
                            f"Null: {'YES' if column[3] == 'YES' else 'NO'}, "
                            f"Default: {column[4]}, "
                            f"Comment: {column[5] or ''}")
        return metadata
    except Error as e:
        st.error(f"Error: {e}")
        return []
    finally:
        if conn:
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
    conn = None
    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        # Convert results to a list of dictionaries
        results_list = []
        for row in results:
            results_list.append(dict(zip(columns, row)))
        
        return results_list
    except Error as e:
        st.error(f"Error executing query: {e}")
        return None
    finally:
        if conn:
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
        if results is not None:
            if results:
                df_results = pd.DataFrame(results)
                st.table(df_results)
            else:
                st.info("The query returned no results.")
        else:
            st.error("Failed to execute the query. Please check the SQL syntax or database connection.")
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"SQL Query: {generated_sql}\n\nResults:\n{df_results.to_string(index=False) if results else 'No results found.'}"
    })

st.sidebar.header("Sample Queries")

sample_queries = [
    "Sales people working in tech companies"
]

for i, query in enumerate(sample_queries):
    st.sidebar.subheader(f"Sample Query {i+1}")
    st.sidebar.text(query)
    