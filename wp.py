import os
import psycopg2
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnablePassthrough
import google.generativeai as genai
from langchain_community.utilities.sql_database import SQLDatabase
import pandas as pd

api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)


conn = psycopg2.connect(
    host=os.getenv("HOST"),
    user=os.getenv("USER"),
    password=os.getenv("PASSWORD"),
    database=os.getenv("DATABASE"),
    port=5432
)

db_uri = f"postgresql+psycopg2://{os.getenv('USER')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}:5432/{os.getenv('DATABASE')}"
db = SQLDatabase.from_uri(db_uri, view_support=True)

def get_table_names(db, allowed_tables=None):
    """
    Extrae los nombres de las tablas utilizables de la base de datos, limitando a las tablas permitidas.
    """
    tables_info = {}
    all_table_names = db.get_usable_table_names()
    
    if allowed_tables:
        table_names = [table for table in all_table_names if table in allowed_tables]
    else:
        table_names = all_table_names

    print("Tablas permitidas después del filtro:", table_names)

    try:
        all_tables_info = db.get_table_info(table_names)
        
        for table in table_names:
            print(f"Procesando información de la tabla: {table}")
            table_info = db.get_table_info([table])
            
            columns = [
                {"column_name": col.split(" ")[0], "data_type": " ".join(col.split(" ")[1:])}
                for col in table_info.split("\n")[1:] if col.strip()
            ]
            tables_info[table] = columns

    except Exception as e:
        print(f"Error al obtener la información de las tablas: {e}")

    return tables_info


allowed_tables = ["llm_fact_ms_drg_test"]
table_names = get_table_names(db,allowed_tables=allowed_tables)


total_tokens = 0
total_llm_calls = 0

def get_gemini_reply(question, prompt):
    """
    Función para generar una respuesta con Gemini.
    - question: pregunta del usuario.
    - prompt: contexto para el modelo.
    """

    global total_tokens, total_llm_calls

    model = genai.GenerativeModel('gemini-1.5-flash')#models/gemini-1.5-pro-latest')
    
    token_count_prompt = model.count_tokens(prompt).total_tokens
    token_count_question = model.count_tokens(question).total_tokens

    response = model.generate_content(f"{prompt}\n\n{question}")

    token_count_response = model.count_tokens(response.text).total_tokens

    total_tokens += token_count_prompt + token_count_question + token_count_response
    total_llm_calls += 1

    return response.text


def get_gemini_reply_sql(question, prompt):
    """
    Función para generar una respuesta con Gemini.
    - question: pregunta del usuario.
    - prompt: contexto para el modelo.
    """

    global total_tokens, total_llm_calls

    model = genai.GenerativeModel('tunedModels/sqleg-8nrr5fqw76yj')#'gemini-1.5-flash')#models/gemini-1.5-pro-latest')
    
    token_count_prompt = model.count_tokens(prompt).total_tokens
    token_count_question = model.count_tokens(question).total_tokens

    response = model.generate_content(f"{prompt}\n\n{question}")

    token_count_response = model.count_tokens(response.text).total_tokens

    total_tokens += token_count_prompt + token_count_question + token_count_response
    total_llm_calls += 1

    return response.text

def get_relevant_tables(question, table_names):
    """
    Identifica las tablas relevantes basándose en la pregunta y el esquema.
    """
    prompt = f"""
    Return the names of ALL the SQL tables that MIGHT be relevant to the user question. 
    The tables are:

    {table_names}

    Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed.

    Important: Only return the name of the table, otherwise return None
    User question: {question}
    """
    #response = get_gemini_reply(question, prompt) # TODO uncommnet when needed
    #return response.split("\n")  # Devuelve una lista de tablas relevantes#TODO uncomment when needed
    
    response=['llm_fact_ms_drg_test', '']#TODO 
    response = [item for item in response if item.strip()]
    return response


def validate_single_query(sql_query):
    """
    Verifica si el modelo generó más de una consulta SQL.
    """
    queries = sql_query.split(";")
    # Eliminar consultas vacías
    queries = [query.strip() for query in queries if query.strip()]
    return queries[0] if queries else "Error: No valid query generated."



def generate_sql_query(question, relevant_tables_info):
    relevant_tables_context = "\n".join(
        f"Table: {table}\nColumns: {', '.join([f'{col['column_name']} ({col['data_type']})' for col in info])}"
        for table, info in relevant_tables_info.items()
    )

    prompt = f"""
    You are an expert POSTGRESQL query generator specializing in pricing transparency, procedure codes, and hospital-related data.

    ### Task:
    Generate **exactly one SQL query** based on the provided question and the relevant tables below. The query must be accurate, clear, and directly address the user's question.

    ### Rules:
    1. **Output Formatting**:
       - Return only one SQL query as output. Do not include any additional text, comments, or delimiters (e.g., no `'`, `"`, or `sql`).
    2. **Focus**:
       - Ensure the query strictly addresses the user's question using only the provided tables and columns.
    3. **Priority**:
       - If there are multiple ways to answer the question, choose the most relevant and concise query.
    4. **Restrictions**:
       - Use only the tables and columns listed in the relevant tables section.
       - Avoid including rows with null values from any table. e.g. `WHERE column1 NOTNULL`
    5. **Pattern Matching**:
       - Use `LIKE` with `%` and `lower()` for pattern matching (e.g., `WHERE lower(column_name) LIKE '%value%'`).
    6. **Error Handling**:
       - If the question is unclear or insufficient data is available, return this message:
         `"Unable to generate query: Question is unclear or no relevant data available."`
    7. **Pricing Column**:
       - The column `standard_charge_negotiated_dollar` represents procedure prices/charges and is the primary column for pricing-related queries.


    ### Relevant Tables:
    {relevant_tables_context}

    ### User Question:
    {question}

    ### SQL Query:
    """
    
    sql_query = get_gemini_reply_sql(question, prompt)
    return sql_query.strip()



# def generate_sql_query(question, relevant_tables_info):
#     relevant_tables_context = "\n".join(
#         f"Table: {table}\nColumns: {', '.join([f'{col['column_name']} ({col['data_type']})' for col in info])}"
#         for table, info in relevant_tables_info.items()
#     )

#     prompt = f"""
#     You are an expert SQL query generator, focusing on pricing transparency, procedure codes, and hospital-related data. 

#     Based on the following relevant tables, generate a **single SQL query** to answer the provided question. Follow these rules and guidelines:

#     ### Query Rules:
#     1. **Output Formatting**:
#     - Return only the SQL query without any additional text, prefixes, or delimiters (e.g., no `'`, `"`, `sql`, or code blocks).
#     2. **Question Focus**:
#     - Base your query strictly on the user's question, including only the necessary information for the response.
#     3. **Table Restrictions**:
#     - Use only the tables and columns provided in the relevant tables list below. Do not reference any table or column outside this list.
#     4. **Pattern Matching**:
#     - Use `LIKE` with `%` and `lower()` for flexible pattern matching (e.g., `WHERE lower(column_name) LIKE '%value%'`).
#     5. **Null Handling**:
#     - Avoid including rows with null values from any table.
#     6. **Grouping**:
#     - When using `GROUP BY`, ensure all selected columns not aggregated are included in the `GROUP BY` clause.
#     - For example: `SELECT column1, column2, AVG(column3) FROM table_name GROUP BY column1, column2`.
#     7. **Aggregations**:
#     - If using aggregations like `AVG` or `SUM` in a `HAVING` or `WHERE` clause, ensure they are also included in the `SELECT` clause.
#     - Example: `SELECT column1, AVG(column2) FROM table_name GROUP BY column1 HAVING AVG(column2) > 100`.
#     8. **Pricing Column**:
#     - The column `standard_charge_negotiated_dollar` represents procedure prices/charges and is the primary column for pricing-related queries.
#     9. **Unclear Questions**:
#     - If the user's question is unclear or lacks sufficient detail, return the message: 
#         **"Unable to generate query: Question is unclear."**
#     10. **Missing Data**:
#         - If no relevant table matches the user's query, return the message:
#         **"Unable to generate query: No relevant data available in the database."**

#     ### Relevant Tables and Columns:
#     {relevant_tables_context}

#     ### User Question:
#     {question}

#     ### SQL Query:
#     """

#     sql_query = get_gemini_reply_sql(question, prompt)
#     return sql_query.strip()


def serialize_result(result):
    if not result or len(result) == 0:
        return "No results found."
    
    try:
        formatted_result = "\n".join(
            [", ".join(map(str, row)) for row in result]
        )
        return formatted_result
    except Exception as e:
        print(f"Error serializando el resultado: {e}")
        return "Serialization error."




def process_question(question):
    print("\n[1] tablas relevantes...")
    relevant_tables = get_relevant_tables(question, list(table_names.keys()))
    print(relevant_tables)
    relevant_tables_info = {table: table_names[table] for table in relevant_tables if table in table_names}
        
    print("\n[2] Generando consulta SQL...")
    sql_query = generate_sql_query(question, relevant_tables_info)

    # Validar y obtener el primer query válido
    sql_query_validate = validate_single_query(sql_query)

    if "Error" in sql_query_validate:
        print("Invalid SQL")
        print(sql_query)  # Muestra el error completo

        print("\n[3] Unable to execute query...")
        return None, "Invalid query","No results"
    
    sql_query = sql_query_validate  # Asegurarse de usar solo el primer query válido
    print("Consulta SQL válida:", sql_query)

    print("\n[3] Ejecutando consulta en la base de datos...")
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql_query)
            col_names = [desc[0] for desc in cursor.description]
            result = cursor.fetchall()
            if not result:
                print("Consulta ejecutada pero no devolvió resultados.")
                return pd.DataFrame(), sql_query ,"No results"
            df = pd.DataFrame(result, columns=col_names)  # Guardar resultados en DataFrame
            print("Resultados de la consulta:", df)
        return df, sql_query, result
    except Exception as e:
        print(f"Error al ejecutar la consulta: {e}")

        return None, f"Error executing query: {str(e)}","No results"




    # sql_query = generate_sql_query(question, relevant_tables_info)
    # sql_query_validate = validate_single_query(sql_query)
    # if "Error" in sql_query_validate:
    #     print("Invalid SQL")
    #     print(sql_query)  # Muestra el error
    #     result = "No data to be retrieved"
    #     sql_query = "Invalid query"
    #     print("\n[3] Unable to execute query...")
    #     return result, sql_query
    # else:
    #     print("Consulta SQL válida:", sql_query)

    #     print("\n[3] Ejecutando consulta en la base de datos...")
    #     try:
    #         with conn.cursor() as cursor:
    #             cursor.execute(sql_query)
    #             result = cursor.fetchall()
    #     except:
    #         result = "No data to be retrieved"
    #     print("Resultados de la consulta:", result)
    #     return result, sql_query


answer_template = """ Given the following question, corresponding Query and SQL result, answer the user question: Question: {question} Query: {sql_query} Result: {result} Answer:"""
answer_prompt = PromptTemplate.from_template(answer_template)


output_parser = StrOutputParser()

def build_chain(question, sql_query, result):
    """
    Construye la cadena de procesamiento.
    """
    chain = (
        RunnablePassthrough()
        | answer_prompt        
        | output_parser       
    )
    return chain


def process_answer(question, sql_query, result):
    """
    Procesa la respuesta final para el usuario.
    """
    # Serializar el resultado antes de pasarlo al prompt
    result_text = serialize_result(result)

    question = str(question).strip()
    sql_query = str(sql_query).strip()
    result_text = str(result_text).strip()

    print("\nEntradas para el modelo:")
    print(f"Pregunta: {question}")
    print(f"Consulta SQL: {sql_query}")
    print(f"Resultado:\n{result_text}")

    generated_prompt = {
        "question": question,
        "sql_query": sql_query,
        "result": result_text
    }

    print("\nPrompt generado para el modelo (diccionario):")
    for key, value in generated_prompt.items():
        print(f"{key}: {value} (tipo: {type(value)})")

    chain = build_chain(question, sql_query, result_text)

    try:
        response = chain.invoke(generated_prompt)
        return response
    except Exception as e:
        print(f"Error al invocar el modelo: {e}")
        raise


def process_answer_direct(question, sql_query, result):
    result_text = serialize_result(result)

    prompt = f"""
    You are a professional analyst specializing in pricing transparency, medical procedure codes, and hospital-related topics. Your expertise includes understanding and interpreting data related to healthcare pricing, hospital billing practices, and procedural standards.

    Given the following question, corresponding SQL query, and result, provide a clear, concise, and accurate answer to the user's question:
    - **Question**: {question}
    - **SQL Query**: {sql_query}
    - **Result**: {result_text}

    ### Instructions:
    1. If the provided question is within your area of expertise, ensure the answer is detailed, actionable, and contextually relevant.
    2. If the question falls outside your area of expertise, politely redirect the user to focus on topics within your domain (e.g., pricing transparency, hospital billing, or procedure codes).
    3. Always maintain a professional tone and avoid making assumptions beyond the provided data.
    4. If **SQL Query** or **Result** doesn't return relevant data, do your best to give the best explanation without negative answers or providing to many deatils about what happening while we attempt to access database.
    5. Never provide information of tables, columns or the query used to retrieve data.

    Your answer should directly address the question and align with the context of pricing transparency and hospital-related information.

    **Answer**:
    """
    try:
        response = get_gemini_reply(question="-", prompt=prompt) 
        print(response)
        return response
    except Exception as e:
        print(f"Error al invocar el modelo directamente: {e}")
        raise




import streamlit as st

logo = r"Correlate-Logo-Final.png"
st.set_page_config(page_title="CorrelateIQ", page_icon=logo, layout="centered")
st.header("Chat with CorrelateIQ")
st.image(logo, width=400)

with st.form(key='question_form'):
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_area(
            "Type your question here:",
            placeholder="Ask a question related to pricing transparency...",
            key="question_input",
            height=100
        )
    with col2:
        submit = st.form_submit_button("Submit")

if submit:
    st.subheader("Processing your query...")

    df, sql_query, result = process_question(question)

    # final_answer = process_answer_direct(question, sql_query, result)
    
    # st.subheader("Answer:")
    # st.write(final_answer)
    # st.subheader("Query Results:")
    # st.dataframe(result)


    if sql_query == "Invalid query":
        st.warning("The query could not be generated. Please refine your question.")
        final_answer = "The query could not be processed due to insufficient or unclear data."
    elif df is None:
        st.error("An error occurred while executing the query. Please try again.")
        final_answer = "No valid data could be retrieved."
    elif df.empty:
        st.warning("The query executed successfully, but no results were returned.")
        final_answer = "No results were found for the given query."
    else:
        try:
            final_answer = process_answer_direct(question, sql_query, result)
        except Exception as e:
            final_answer = f"An error occurred while generating the answer: {e}"

    # Mostrar la respuesta final al usuario
    st.subheader("Answer:")
    st.write(final_answer)    

    if isinstance(df, pd.DataFrame) and not df.empty:
        st.subheader("Query Results:")
        st.dataframe(df)



st.sidebar.title("Usage Stats")
st.sidebar.markdown(f"**Total Tokens Used:** {total_tokens}")
st.sidebar.markdown(f"**Total LLM Calls:** {total_llm_calls}")

if sql_query and sql_query != "Invalid query":
    st.sidebar.markdown(f"**Query Generated:** {sql_query}")
else:
    st.sidebar.markdown("**Query Generated:** None or invalid")



# question = "List the maximum price per service line."
# result, sql_query = process_question(question)


# print("\nResultado SQL:", result)


# final_answer = process_answer_direct(question, sql_query, result)
# print("\nRespuesta final al usuario:")
# print(final_answer)

# print("\nResumen Final:")
# print(f"Total de Tokens Utilizados: {total_tokens}")
# print(f"Total de Llamadas al Modelo LLM: {total_llm_calls}")




