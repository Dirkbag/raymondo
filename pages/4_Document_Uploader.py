import streamlit as st
import os
import time
import tempfile
from supabase.client import Client, create_client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Document Uploader",
    page_icon="📄",
    layout="wide"
)
st.title("📄 Document Uploader")
st.markdown("""
Use this page to upload new PDF documents to Raymondo's knowledge base. 
This updated version ensures document chunks are correctly linked to their source file.
""")

# --- 2. Centralized Resource Initialization ---
@st.cache_resource
def get_resources():
    """Initializes all necessary connections and models once and caches them."""
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_SERVICE_KEY"]
    supabase: Client = create_client(supabase_url, supabase_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.secrets["OPENAI_API_KEY"])
    vector_store = SupabaseVectorStore(
        embedding=embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
        chunk_size=1000,
    )
    return supabase, vector_store

# --- 3. Function to Process a Single PDF File ---
def process_and_ingest_pdf(file_uploader_object, supabase_client, vector_store):
    """Processes a single uploaded PDF file and ingests it into the vector store with correct metadata."""
    file_name = file_uploader_object.name
    
    # Check for duplicates first
    response = supabase_client.table("uploaded_documents").select("name").eq("name", file_name).execute()
    if response.data:
        st.warning(f"⚠️ Skipping duplicate file: `{file_name}`")
        return

    st.info(f"⏳ Processing `{file_name}`...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_uploader_object.getbuffer())
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # --- THIS IS THE CRITICAL FIX ---
        # Override the temporary path in metadata with the original filename
        for doc in documents:
            doc.metadata['source'] = file_name
        
        author = documents[0].metadata.get("author", "Unknown") if documents else "Unknown"

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        if not docs:
            st.warning(f"⚠️ Could not extract any text from `{file_name}`. Skipping.")
            return

        # Ingest into Vector Store
        vector_store.add_documents(docs)
        st.info(f"  - Ingested {len(docs)} chunks for `{file_name}`.")

        # Record Successful Upload
        supabase_client.table("uploaded_documents").insert({"name": file_name, "author": author}).execute()
        st.success(f"✅ Successfully processed and ingested `{file_name}`.")

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# --- 4. Main Streamlit UI ---
try:
    supabase, vector_store = get_resources()

    uploaded_files = st.file_uploader(
        "Choose PDF files to add to the knowledge base",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write("Files to be uploaded:")
        for file in uploaded_files:
            st.markdown(f"- `{file.name}`")

        if st.button("Start Upload and Processing", type="primary"):
            with st.spinner("Processing files... This may take a few minutes."):
                for file in uploaded_files:
                    process_and_ingest_pdf(file, supabase, vector_store)
            
            st.balloons()
            st.header("🎉 All files have been processed!")
            st.info("Navigate to the 'Document Management' page to see the updated chunk counts.")

except Exception as e:
    st.error("An error occurred during the setup.")
    st.exception(e)

