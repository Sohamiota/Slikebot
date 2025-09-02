import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

class VectorDB:
    def __init__(
        self,
        pdf_path: str,
        db_location: str = "./chroma_langchain_db",
        collection_name: str = "slike_reviews",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.pdf_path = pdf_path
        self.db_location = db_location
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self):
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        documents = splitter.split_documents(pages)

        vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.db_location,
            embedding_function=self.embeddings
        )

        if len(vector_store.get()["ids"]) == 0 and documents:
            ids = [str(i) for i in range(len(documents))]
            vector_store.add_documents(documents=documents, ids=ids)

        return vector_store

    def get_retriever(self, k: int = 5):
        return self.vector_store.as_retriever(search_kwargs={"k": k})

pdf_path = "Live Streaming Platform User Guide.pdf"
vector_db = VectorDB(pdf_path)
retriever = vector_db.get_retriever(k=5)
