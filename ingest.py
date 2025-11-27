from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

MODEL = "gpt-4.1-nano"
DB_NAME = str("vector_db")

load_dotenv(override=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def get_docs():
    loader = PyPDFLoader(file_path="data/dataset.pdf")
    docs = loader.load()
    return docs


def create_chanks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chanks = text_splitter.split_documents(docs)
    return chanks


def create_embeddings(chanks):
    if os.path.exists(DB_NAME):
        Chroma(
            persist_directory=DB_NAME, embedding_function=embeddings
        ).delete_collection()

    vectorstore = Chroma.from_documents(chanks, embeddings, persist_directory=DB_NAME)
    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(
        f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store"
    )
    return vectorstore


if __name__ == "__main__":
    docs = get_docs()
    chanks = create_chanks(docs)
    create_embeddings(chanks)
