"""RAG (Retrieval-Augmented Generation) System Implementation."""

import os
from typing import List, Dict, Optional
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


class RAGPipeline:
      """Retrieval-Augmented Generation pipeline for knowledge-grounded responses."""

    def __init__(
              self,
              api_key: str = None,
              pinecone_key: str = None,
              model: str = "gpt-3.5-turbo"
    ):
              """Initialize RAG pipeline.

                              Args:
                                          api_key: OpenAI API key
                                                      pinecone_key: Pinecone API key
                                                                  model: LLM model to use
                                                                          """
              self.api_key = api_key or os.getenv("OPENAI_API_KEY")
              self.pinecone_key = pinecone_key or os.getenv("PINECONE_API_KEY")
              self.model = model
              self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
              self.llm = OpenAI(openai_api_key=self.api_key, model_name=model)
              self.qa_chain = None

    def load_documents(self, file_path: str) -> List[str]:
              """Load documents from file."""
              try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                              content = f.read()
                                          return [content]
except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []

    def create_vectorstore(self, documents: List[str], index_name: str = "rag-index"):
              """Create vector store from documents."""
              splitter = RecursiveCharacterTextSplitter(
                  chunk_size=1000,
                  chunk_overlap=200
              )
              texts = splitter.split_documents(documents)
              self.vectorstore = Pinecone.from_documents(
                  texts,
                  self.embeddings,
                  index_name=index_name
              )

    def build_qa_chain(self):
              """Build QA chain from vector store."""
              if not self.vectorstore:
                            raise ValueError("Vector store not initialized")
                        self.qa_chain = RetrievalQA.from_chain_type(
                                      llm=self.llm,
                                      chain_type="stuff",
                                      retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
                        )

    def query(self, question: str) -> Dict[str, str]:
              """Query the RAG pipeline.

                              Args:
                                          question: User question

                                                              Returns:
                                                                          Dictionary with question and answer
                                                                                  """
        if not self.qa_chain:
                      raise ValueError("QA chain not built. Call build_qa_chain first.")
                  result = self.qa_chain({"query": question})
        return {"question": question, "answer": result["result"]}


if __name__ == "__main__":
      # Example usage
      pipeline = RAGPipeline()
    response = pipeline.query("What is machine learning?")
    print(f"Q: {response['question']}")
    print(f"A: {response['answer']}")
