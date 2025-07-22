from supabase import create_client
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
# import streamlit
import streamlit as st
import os

# ✅ Connect to Supabase
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_SERVICE_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize Supabase client
supabase = create_client(supabase_url, supabase_key)

# ✅ Load completions data
data = supabase.table("completions").select("*").execute().data
df = pd.DataFrame(data)

# ✅ Helper: Build a natural language summary of each row
def row_to_summary(row):
    return (
        f"Client: {row['Client']}, Adviser: {row['Adviser']}, Source: {row['Source']}, "
        f"Product: {row['Product']}, Lender: {row['Lender']}, Release: £{row['Release']}, "
        f"Interest Rate: {row['Interest Rate']}%, "
        f"Completion Date: {row.get('Comp Date')}, Cancelled: {row.get('Cancel Date')}, "
        f"Proc Fee: £{row['Proc Fee']}, Advice Fee: £{row['Adv Fee']}, MF: {row['MF']}"
    )

# ✅ Convert to LangChain Document format
documents = []
for _, row in df.iterrows():
    content = row_to_summary(row)
    metadata = {
        "client": row["Client"],
        "adviser": row["Adviser"],
        "product": row["Product"],
        "completed": row.get("Comp Date"),
        "source": "completions"
    }
    documents.append(Document(page_content=content, metadata=metadata))

# ✅ Initialize vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = SupabaseVectorStore(
    client=supabase,
    table_name="completions_embedded",
    query_name="match_documents",
    embedding=embeddings
)

# ✅ Embed and upload
vector_store.add_documents(documents)

print(f"✅ Successfully embedded and uploaded {len(documents)} completions to Supabase.")
