import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
from models import Models

load_dotenv()

# Initialize LLMs
models = Models()
embeddings = models.embeddings_hf
llm = models.chat_model

# Constants
data_folder = "./data"
chunk_size = 1000
chunk_overlap = 50
check_interval = 10 # checking interval for ingestion

# Define chroma vector DB
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,#embedding model
    persist_directory="./db/chroma.db",
)

# Ingestion of PDFs (utility function)
def ingest_files(fp):
    if not fp.lower().endswith('.pdf'):
        print(f"Skipped a non-PDF file : {fp}")
        return  

    print(f"PDF Ingestion Started for : {fp}")
    loader = PyPDFLoader(fp)
    loaded_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n",
        " ", ""]
    )

    documents = text_splitter.split_documents(loaded_documents)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    print(f"Adding {len(documents)} documents to the vector DB..")
    vector_store.add_documents(documents=documents, ids=uuids)
    print(f"Done ingesting {fp}")

def main():
    while True:
        for filename in os.listdir(data_folder):
            if not filename.startswith("_"): # Prevent ingesting same file twice
                fp = os.path.join(data_folder, filename)
                ingest_files(fp)
                new_fn = "_" + filename
                new_fp = os.path.join(data_folder, new_fn)
                os.rename(fp, new_fn) # Rename once file gets ingested
        time.sleep(check_interval) # Check data folder for new files

if __name__ == "__main__":
    main()