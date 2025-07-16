# import basics
import os
import logging
from dotenv import load_dotenv



# UNCOMMENT THE BELOW FOR CHECKING
# print("Now in directory:", os.getcwd())
# print("Current working directory:", os.getcwd())
# print("Directory exists:", os.path.isdir("documents"))
# print("Files in folder:", os.listdir("documents"))

# import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

# import supabase
from supabase.client import Client, create_client

# Force working directory to this script's folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
# ABSOLUTE PATH: os.chdir(r"C:\Users\DerekHenderson\OneDrive - Retirement Solutions\Documents\Python_scripts\Raymondo_RAG")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load environment variables
load_dotenv()  

# initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# load pdf docs from folder 'documents'
loader = PyPDFDirectoryLoader("documents")

# split the documents in multiple chunks
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# store chunks in vector store
vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    chunk_size=1000,
)

print(f"Loaded {len(docs)} document chunks.")
