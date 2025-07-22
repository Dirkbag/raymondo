import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.utilities import SQLDatabase
from supabase import create_client
from sqlalchemy import create_engine
import re

# --- 1. Page Setup ---
st.set_page_config(
    page_title="🦜 Raymondo Chatbot",
    page_icon="🦜",
    layout="wide"
)
st.title("🦜 Raymondo Chatbot")

# --- Temporary Debugging Section ---
# This section will help us see which secrets the app can access on Streamlit Cloud.
with st.expander("🔍 Secrets Debug Information"):
    st.write("Checking for available secrets...")
    try:
        all_secrets = st.secrets.to_dict()
        if not all_secrets:
            st.warning("Warning: `st.secrets` is empty.")
        else:
            st.success("Secrets dictionary is available.")
            st.write("The following secret keys were found:")
            # We only print the keys (names) for security, not the values.
            st.write(list(all_secrets.keys()))
    except Exception as e:
        st.error(f"An error occurred while accessing st.secrets: {e}")
# --- End of Debugging Section ---


with st.expander("ℹ️ How to use Raymondo (click to expand)"):
    st.markdown("""
    ### 👋 Welcome to Raymondo — Your AI Chat Assistant
    This tool helps Retirement Solutions staff retrieve internal knowledge using natural language.
    #### 📌 How to Use:
    1.  **Select the data source** you want to query using the radio buttons.
        - **Internal Documents (PDFs)**: For searching training manuals, guides, and other PDF documents.
        - **Case Data (Completions)**: For querying specific financial data from the completions table.
    2.  Type a question into the chat box below.
    
    #### 🔐 Access & Setup:
    - You must be signed in with an authorised email to use this tool.
    - **For SQL Agent to work**, the `DB_HOST`, `DB_USER`, `DB_PASSWORD`, and `DB_NAME` secrets must be set.
    - If you encounter access issues, contact: `derek.henderson@retirementsolutions.co.uk`
    ---
    """)

# --- Helper function to check for DB secrets ---
def check_db_secrets():
    """Checks if all required database secrets are present in Streamlit secrets."""
    return all(s in st.secrets for s in ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_NAME"])

# --- 2. System Initialization ---
# @st.cache_resource runs this function only once per session, making the app fast.
@st.cache_resource
def initialize_system():
    """
    Initializes all necessary connections, models, and agents.
    Gracefully handles the absence of SQL database credentials.
    """
    # --- General Setup ---
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings()
    supabase_client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_SERVICE_KEY"])

    # --- Agent 1: Document Retriever Agent ---
    doc_vector_store = SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
    doc_retriever = doc_vector_store.as_retriever()
    doc_tool = create_retriever_tool(
        retriever=doc_retriever,
        name="internal_document_search",
        description="Search for information from the internal knowledge base of PDF documents."
    )
    doc_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Raymondo, an expert assistant who answers user questions based on retrieved information from internal documents."),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    doc_agent = create_tool_calling_agent(llm, [doc_tool], doc_prompt)
    doc_executor = AgentExecutor(agent=doc_agent, tools=[doc_tool], verbose=True, handle_parsing_errors=True)

    # --- Agent 2: SQL Agent ---
    sql_executor = None
    if check_db_secrets():
        try:
            db_uri = f"postgresql://{st.secrets['DB_USER']}:{st.secrets['DB_PASSWORD']}@{st.secrets['DB_HOST']}:5432/{st.secrets['DB_NAME']}"
            db_engine = create_engine(db_uri)
            db = SQLDatabase(db_engine, include_tables=['completions'])
            sql_executor = create_sql_agent(
                llm=llm,
                db=db,
                agent_type="openai-tools",
                verbose=True,
                handle_parsing_errors=True
            )
        except Exception as e:
            print(f"SQL Agent Initialization Error: {e}")
            
    return doc_executor, sql_executor

# --- Initialization & State Check ---
doc_executor, sql_executor = initialize_system()
is_sql_agent_available = sql_executor is not None

# --- 3. Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
# --- UI for Data Source Selection ---
st.markdown("---")

radio_options = ["Internal Documents (PDFs)"]
if is_sql_agent_available:
    radio_options.append("Case Data (Completions)")
else:
    st.warning(
        "**SQL Agent is unavailable.** To enable querying of 'Case Data', please set the required database secrets in your configuration.",
        icon="🔥"
    )

data_source = st.radio(
    "**Select Data Source to Query**",
    options=radio_options,
    horizontal=True
)

# --- Chat Input and Processing ---
user_input = st.chat_input(f"Ask Raymondo about {data_source}...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    ai_response = ""
    with st.chat_message("assistant"):
        if data_source == "Internal Documents (PDFs)":
            if not doc_executor:
                st.error("Document Agent is not available.")
            else:
                with st.spinner("Raymondo is searching internal documents..."):
                    result = doc_executor.invoke({"input": user_input})
                    ai_response = result["output"]
                    st.markdown(ai_response)

        elif data_source == "Case Data (Completions)":
            if not sql_executor:
                st.error("SQL Agent is not available.")
            else:
                with st.spinner("Raymondo is querying the case database..."):
                    prompted_input = f"Based on the 'completions' table, answer the user's question: {user_input}"
                    result = sql_executor.invoke({"input": prompted_input})
                    ai_response = result["output"]
                    st.markdown(ai_response)
    
    if ai_response:
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
