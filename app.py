import streamlit as st
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

# Initialize the LLM (Ollama)
llm = Ollama(model="mistral")  # Change to your preferred model

# Load and create the vectorstore
def load_vectorstore():
    loader = TextLoader("knowledge_base.txt")  # Make sure this file is in your repo
    index = VectorstoreIndexCreator(
        embedding=OllamaEmbeddings()
    ).from_loaders([loader])
    return index.vectorstore.as_retriever()

retriever = load_vectorstore()

# Create a RAG-powered chatbot
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Streamlit UI
st.title("🛍️ AI Sales Assistant Chatbot")
st.markdown("Ask me anything about our products!")

user_query = st.text_input("Enter your query:", "")

if st.button("Ask"):
    if user_query:
        response = qa_chain.run(user_query)
        st.write("💬 Chatbot:", response)
    else:
        st.warning("Please enter a query.")