import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter

load_dotenv()

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

MODEL = "gemma:7b"
model = Ollama(model = MODEL)
embeddings = OllamaEmbeddings()
output_parser = StrOutputParser()
urls = ["https://medium.com/@deepanshut041/introduction-to-surf-speeded-up-robust-features-c7396d6e7c4e", "https://www.freecodecamp.org/news/beginners-guide-to-langchain/"]
chunks = convert_url_to_documents(urls)

vectorstore = DocArrayInMemorySearch.from_documents(chunks, embedding=embeddings)
retriever = vectorstore.as_retriever()


template = """
Answer the question based on the context below. Provide a detailed answer that relates to the question based on the context. If you cannot answer the question, reply that you dont know or that you need more context.

Context: {context}

Question: {question}

"""
prompt = PromptTemplate.from_template(template)

chain = (
    {"context": itemgetter ("question") | retriever, "question": itemgetter ("question")}
        | prompt
        | model
        | output_parser)
