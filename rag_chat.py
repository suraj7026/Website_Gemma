import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from functools import cache


st.set_page_config(layout="wide")

@cache
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    MODEL = "gemma:7b"
    model = ChatOllama(model=MODEL)
    embeddings = OllamaEmbeddings()
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    output_parser = StrOutputParser()
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def extract_text_with_selenium(url):
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        service = Service(executable_path='/usr/local/bin/chromedriver')
        driver = webdriver.Chrome(service=service, options=chrome_options)

        driver.get(url)
        driver.implicitly_wait(10)
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        metadata = {"source": url}
        document = Document(page_content=text, metadata=metadata)
        return document
    except Exception as e:
        return None
    finally:
        driver.quit()

def convert_url_to_documents(urls):
    documents = [extract_text_with_selenium(url) for url in urls if extract_text_with_selenium(url)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    return chunks

# def generate_response(user_question):
#     embeddings = OllamaEmbeddings()
#     new_db = FAISS.load_local("faiss_index", embeddings)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"context": docs, "question": user_question}, return_only_outputs=True)
#     return response


def generate_response(user_question):
    embeddings = OllamaEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    # st.write(docs)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]



def get_vector_store(text_chunks):
    texts = [chunk.page_content for chunk in text_chunks]  # Extract text from Document objects
    embeddings = OllamaEmbeddings()
    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings)
    vector_store.save_local("faiss_index")

with st.sidebar:
    urls_input = st.text_area("Enter URLs (one per line):")
    urls = list(set(urls_input.split("\n")))

    if st.button("Create Memory of Websites"):
        with st.spinner("Processing..."):
            chunks = convert_url_to_documents(urls)
            get_vector_store(chunks)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = generate_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
