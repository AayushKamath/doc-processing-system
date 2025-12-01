import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from html_template import css,bot_template,user_template

import os
openai_key = os.getenv("OPENAI_API_KEY")


def get_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(raw_txt):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_txt)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts = text_chunks,  embedding = embeddings)
    return vectorstore

def get_conv_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages = True)
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conv_chain

def handle_input(user_q):
    response = st.session_state.conversation({'question': user_q})
    st.session_state.chat_history = response['chat_history']




def main():
    load_dotenv()
    st.set_page_config(page_title="Document Lookup", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Document Lookup :books:")

    with st.sidebar:
        st.subheader("Documents")
        docs = st.file_uploader("Upload your pdf here", accept_multiple_files=True)
        
        if st.button('Upload'):
            with st.spinner("Processing"):
                # Raw contents of file
                raw_txt = get_text(docs)

                # Create chunks of text
                text_chunks = get_chunks(raw_txt)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Conversation
                st.session_state.conversation = get_conv_chain(vectorstore)
                st.session_state.chat_history = []
    


    for msg in st.session_state.chat_history:
        if hasattr(msg, "type"):
            role = msg.type  # For newer versions
        elif hasattr(msg, "role"):
            role = msg.role  # For older versions
        else:
            role = "assistant"  # Default fallback

        template = user_template if role == "human" else bot_template
        st.write(template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)


    # Input at bottom with on_change callback
    def submit_query():
        user_q = st.session_state.pending_query
        if st.session_state.conversation and user_q:
            response = st.session_state.conversation({'question': user_q})
            st.session_state.chat_history = response['chat_history']
            st.session_state.pending_query = ""  # Clear after processing

    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)

    st.text_input("Type your message…", key="pending_query", on_change=submit_query)


if __name__ == '__main__':
    main()