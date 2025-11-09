import os # used for file and directory operations (like creating folders).
from pathlib import Path # an object-oriented way to handle file paths (better than raw strings).
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader # used to read PDF files and extract text + metadata.

def process_all_pdfs(pdf_directory: str): # we pass the path of the folder  like "Data/"
    """
    Process all PDF files in a directory.
    Returns a list of Document objects from LangChain.
    """
    all_documents = []  # place holder to keep all the pdf documents that are inside "Data/" directory
    pdf_dir = Path(pdf_directory) # Converts your folder path (like "Data/") into a Path object
    pdf_files = list(pdf_dir.glob("**/*.pdf")) #finds all PDFs recursively (even in subfolders)
    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))  # is a document loader that helps you read a PDF file and extract its text and metadata into a format LangChain can understand.
            documents = loader.load() #returns a list of Document objects


            # For each document (page), adds extra metadata:
            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
            
            print(f"Loaded {len(documents)} pages")
        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents
