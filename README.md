## Text to Query using Llama-Index

### Features

- Given a user_prompt we generate sql query using that prompt and execute it for the given database.

- First we fetch the table details (schema), of the given database and index this data.

- Then we accept the user_prompt as request body and modify the prompt accordingly with the given table meta data.

- With the updated prompt we generate the sql query using `Gemini`.

- We execute this query in our database to fetch the results.


### Create a virtual environment

```bash
python3 -m venv venv
```


### Activate the virtual environment

```bash
source venv/bin/activate
```


### Installations

```bash
pip install -r requirements.txt
```


### Alternate Installations

```bash
pip install ipykernel
```

```bash
pip install python-dotenv
```


```bash
pip install llama-index
```


```bash
pip install llama-index-llms-gemini
```

```bash
pip install llama-index-embeddings-gemini
```

```bash
pip install mysql-connector-python
```

### Setup for the Project

- Create a `.env` file in the root directory (same path as `main.ipynb` )

- Add the following details 

    - `GEMINI_API_KEY`=<GEMINI_API_KEY>

    - `DB_USER`=<Username_of_the_MySQL_Database>

    - `DB_PASSWORD`=<Password_of_the_MySQL_Database>

    - `DB_HOST`=`localhost`

    - `DB_NAME`=<Name_of_the_database>

### Run the project on Streamlit

```bash
streamlit run main.py
```
