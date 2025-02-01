import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv(dotenv_path='../.env')

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

document_path = "./documents/"
loader = DirectoryLoader(document_path, show_progress=True)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=200,
)
split_documents = text_splitter.split_documents(documents)

# Create embeddings and store in FAISS
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = FAISS.from_documents(split_documents, embeddings)

# Save the FAISS database locally
db.save_local("faiss_db")