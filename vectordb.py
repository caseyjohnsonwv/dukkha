import glob
import os
import re
from typing import List, Tuple
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pypdf
import env
from log import get_logger


LOGGER = get_logger(__name__)


class VectorDB:
    def __init__(self):
        self.vectorstore = FAISS.from_documents(
            documents = self._load_pdf_docs('./dat/*.pdf'),
            embedding = OpenAIEmbeddings(api_key=env.OPENAI_API_KEY, model=env.OPENAI_EMBEDDINGS_MODEL),
        )

    def search(self, query:str, k=8) -> List[Tuple[Document, float]]:
        LOGGER.info(f"Retrieving top {k} similar documents for query: '{query}'")
        return self.vectorstore.similarity_search_with_score(query=query, k=k)


    def _load_pdf_docs(self, pdf_glob_expr:str) -> List[Document]:
        documents = []
        # load pdf files
        pdf_files = glob.glob(pdf_glob_expr)
        LOGGER.info(f"Found {len(pdf_files)} PDF files to index")
        for f in pdf_files:
            filename = os.path.basename(f)
            with open(f, 'rb') as pdf:
                reader = pypdf.PdfReader(pdf)
                LOGGER.info(f"Document '{filename}' has {reader.get_num_pages()} pages to vectorize")
                for page in reader.pages:
                    LOGGER.info(f"Vectorizing page {page.page_number}...")
                    page_text = page.extract_text()
                    page_text_cleaned = re.sub('\s+', ' ', page_text)
                    metadata = {'filename':filename, 'page_number':page.page_number}
                    page_document = Document(page_content=page_text_cleaned, metadata=metadata)
                    documents.append(page_document)
        # split documents with text splitter if needed
        text_splitter = RecursiveCharacterTextSplitter(separators=['\s+'], is_separator_regex=True, keep_separator=True, chunk_size=2000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        LOGGER.info(f"Created {len(docs)} total documents after loading and splitting")
        # return list of documents ready for embedding
        return docs
    