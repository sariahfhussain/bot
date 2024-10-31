import os
import pdfplumber
from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class CustomPDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with pdfplumber.open(self.file_path) as pdf:
            text = ""
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
        return Document(page_content=text.strip(), metadata={"source": self.file_path})

def create_vector_store():
    # Load documents from the "data/" directory using the custom loader
    documents = []
    data_directory = "data/"
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        print(f"Created {data_directory} directory. Please add your PDF files there.")
        return

    # Load PDF documents
    for filename in os.listdir(data_directory):
        if filename.endswith(".pdf"):
            loader = CustomPDFLoader(os.path.join(data_directory, filename))
            documents.append(loader.load())

    if not documents:
        print("No PDF files found in the data directory.")
        return

    print(f"Loading documents... Found {len(documents)} files")

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500
    )
    print("Splitting documents into chunks...")
    text_chunks = text_splitter.split_documents(documents)
    print(f"Created {len(text_chunks)} text chunks")

    # Create and persist vector store
    persist_directory = "doc_db"
    embedding = HuggingFaceEmbeddings()

    print("Creating vector store from documents...")
    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )
    print("Vector store created and saved to disk.")

if __name__ == "__main__":
    create_vector_store()