from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


import os
import faiss
import pandas as pd
from tqdm import tqdm
import glob  # Import glob module
import pyarrow.parquet as pq
import ipywidgets as widgets


# Create a GPT4AllEmbeddings object
embeddings = GPT4AllEmbeddings()
print(embeddings)

def read_vector_store(vector_store_path):
    # Load the vector store
    vector_store = FAISS.load_local(vector_store_path, embeddings)
    return vector_store

def read_index(index_path):
    # Load the FAISS index
    index = faiss.read_index(index_path)
    return index


#load llm model
def load_llm_model():
    llm = Ollama(model="llama2")
    return llm

def get_document_chain(llm, retriever, query):
      # Create a prompt      
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context and metadata:

    <context>
    {context}
    </context>
    <metadata>
    {metadata}
    <metadata>
    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain

def get_relenvant_documents(retriever, query):
    documents = retriever.get_relevant_documents(query)

    return documents



def get_retrieval_chain():
    
        
    llm = load_llm_model()

    vector = read_vector_store('vectorStore')

    # Load the FAISS index
    index = read_index(os.path.join('vectorStore','index.faiss'))
    # Print the number of vectors and their dimensionality
    print(f"Number of vectors in the index: {index.ntotal}")
    print(f"Vector dimensionality: {index.d}")
    
    retriever = vector.as_retriever()


    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", "Act as Expert in Medical field by providing refrences and scientifc conclusions for user's questions based on the below context:\n\n{context} and metadata"),
        ("user", "{input}"),
        ("user", '''Given the above conversation, generate a search query to look up
        in order to get information relevant to the conversation''')
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    return (retrieval_chain, retriever)



#execute this function to get the retrieval chain
#when this file gets executed, the retrieval chain will be created

if __name__ == "__main__":
    chain, retriever = get_retrieval_chain()
    query = "where was the articles published (place of publication PL)?"
    chat_history = [HumanMessage(content=""), AIMessage(content="Answered!")]
    while True:
        query = input("Enter your query (type exit to leave chat): ")
        if query == "exit":
            print("thank you for chatting with Medical Expert RAG System! Hope you have a good time!")
            break
        docs = get_relenvant_documents(retriever, query)
        context = docs[0]
        # print("docs retrieved")
        response = chain.invoke({
            "chat_history": chat_history,
            "input": query,
            "context": context,
        })
        print(response["answer"])
        chat_history = response["chat_history"]




