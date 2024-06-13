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
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores import Chroma
from operator import itemgetter
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
    model = Ollama(model = MODEL)
    embeddings = OllamaEmbeddings()
    # output_parser = StrOutputParser()
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def extract_text_with_selenium(url):
    try:
        # Configure Selenium WebDriver (adjust for your setup)
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        service = Service(executable_path='/usr/local/bin/chromedriver')
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Load the webpage
        driver.get(url)

        # Wait for the page to load (adjust wait time if needed)
        driver.implicitly_wait(10)

        # Get the page source after JavaScript execution
        page_source = driver.page_source

        # Parse the page source with BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')

        # Extract text from the parsed HTML
        text = soup.get_text(separator=' ', strip=True)

        # Create a LangChain Document object
        metadata = {"source": url}
        document = Document(page_content=text, metadata=metadata)
        return document

    except Exception as e:
        # st.error(f"Error extracting text from the webpage: {e}")
        return None

    finally:
        # Close the browser
        driver.quit()

def convert_url_to_documents(urls):

    documents = []
    for url in urls:
        document = extract_text_with_selenium(url)
        if document:
            documents.append(document)


    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    return chunks

# def generate_response(input_text,chain,retriever):
#     # Retrieve the vectorstore and retriever from the session state
#     retriever = st.session_state.get("retriever")
#     if retriever:
#         return chain.invoke({"question": f"{input_text}", "retriever": retriever})
#     else:
#         st.error("Memory of websites has not been created yet.")
#         return None


def generate_response(user_question):
    embeddings = OllamaEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"context": docs, "question": user_question}, return_only_outputs=True)
    return response



def get_context(question):
    return retriever.invoke(question)

def get_vector_store(text_chunks):
    embeddings = OllamaEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


with st.sidebar:
    # Get multiple URLs from the user
    urls_input = st.text_area("Enter URLs (one per line):")
    urls = urls_input.split("\n")  # Split into list
    urls = list(set(urls))

    if st.button("Create Memory of Websites"):
        with st.spinner("Processing..."):
            chunks = convert_url_to_documents(urls)
            get_vector_store(chunks)
        # Create the vectorstore and retriever
            # vectorstore = Chroma.from_documents(documents=chunks, collection_name="rag-chroma", embedding=embeddings)
            # retriever = vectorstore.as_retriever()

        # Store the vectorstore and retriever in session state
        # st.session_state.vectorstore = vectorstore
        # st.session_state.retriever = retriever





# template = """
# Answer the question based on the context below. Provide a detailed answer that relates to the question based on the context. If you cannot answer the question based on the context, tell that no context was found and give your own answer that fits the question best.

# Context: {context}

# Question: {question}

# """


# prompt = PromptTemplate.from_template(template)
# chain = ( {"context": itemgetter ("question") | retriever, "question": itemgetter ("question")} | prompt | model | output_parser)




# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = generate_response(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
