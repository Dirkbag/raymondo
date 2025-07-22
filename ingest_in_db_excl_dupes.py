import os
import time
import logging
from dotenv import load_dotenv
from supabase.client import Client, create_client
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore

# Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Supabase config
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load PDFs
loader = PyPDFDirectoryLoader("documents")
documents = loader.load()

# Deduplicate and record metadata
processed_docs = []
for doc in documents:
    file_name = os.path.basename(doc.metadata.get("source", "unknown.pdf"))
    author = doc.metadata.get("author", "Unknown")

    # Check for duplicates
    response = supabase.table("uploaded_documents").select("name").eq("name", file_name).execute()
    if response.data and len(response.data) > 0:
        logger.info(f"Skipping duplicate: {file_name}")
        continue

    # Insert metadata
    supabase.table("uploaded_documents").insert({
        "name": file_name,
        "author": author
    }).execute()

    processed_docs.append(doc)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(processed_docs)

# Init vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    chunk_size=1000,
)

# Ingest in batches to avoid rate limits
BATCH_SIZE = 100
for i in range(0, len(docs), BATCH_SIZE):
    batch = docs[i:i + BATCH_SIZE]
    try:
        vector_store.add_documents(batch)
        logger.info(f"Ingested batch {i // BATCH_SIZE + 1} with {len(batch)} chunks.")
    except Exception as e:
        logger.warning(f"Failed to insert batch {i // BATCH_SIZE + 1}: {str(e)}")
    time.sleep(2)  # avoid rate-limiting

print(f"✅ Loaded {len(docs)} chunks from {len(processed_docs)} unique PDFs.")