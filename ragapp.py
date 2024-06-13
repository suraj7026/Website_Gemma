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



def get_documents_text(url):
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

def get_url_text_chunks(urls):

    documents = []
    for url in urls:
        document = get_documents_text(url)
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



def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    MODEL =  "gemma:7b"
    llm = ChatOllama(model = MODEL)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# def handle_userinput(user_question):
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']

#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})

    # Assume response['chat_history'] returns a list of messages
    for message in response['chat_history']:
        st.session_state.messages.append({"role": "assistant", "content": message.content})

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple multiple websites")


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.header("Chat with multiple Websites")
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        # Display user message in chat message container
        st.chat_message("user").markdown(user_question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})

        handle_userinput(user_question)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    with st.sidebar:
        st.subheader("Your websites")
        urls_input = st.text_area("Enter URLs (one per line):")
        urls = urls_input.split("\n")  # Split into list
        urls = list(set(urls))
        if st.button("Process"):
            with st.spinner("Processing"):


                # get the text chunks
                text_chunks = get_url_text_chunks(urls)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
