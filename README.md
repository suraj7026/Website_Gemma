# RAG Streamlit Application

This project is a Retrieval-Augmented Generation (RAG) application built using Streamlit. It allows users to input a URL of a website, extract text from the website using Selenium, split the text into chunks, store these chunks in a vector database, and create a conversational AI using LangChain with Google's Gemma model. Users can then interact with the AI by asking queries, which fetch context from the vector database and generate answers.


![Application Overview](https://imgur.com/a/gonJYUe)


## Installation

To run this project, you need to install the following Python modules:

```bash
pip install streamlit
pip install python-dotenv
pip install langchain-community
pip install selenium
pip install beautifulsoup4
pip install faiss-cpu
```
## Modules Used

* Streamlit: For creating the web application interface.
* dotenv: For loading environment variables.
* LangChain Community:
  * `Ollama`: Language model for generating responses.
  * `ChatOllama`: Chat model for interactive conversations.
  * `OllamaEmbeddings`: For embedding documents.
  * `StrOutputParser`: For parsing output strings.
* `PyPDFLoader`: For loading PDF documents.
* Selenium: For web scraping to extract text from websites.
* `webdriver`: To control the web browser.
* `chrome.service.Service`: To manage the Chrome browser service.
* `chrome.options.Options`: To set options for Chrome browser.
* `BeautifulSoup`: For parsing HTML and extracting text.
* `FAISS`: For storing and searching vector embeddings.
* LangChain:
  * `load_qa_chain`: For loading question-answering chains.
  * `Document`: For managing document data.
  * `RecursiveCharacterTextSplitter`: For splitting text into chunks.
  * `PromptTemplate`: For creating prompts.
* functools:
  * `cache`: For caching function results.

## Application Workflow
* User Input: The user provides a URL of a website.
* Text Extraction: Selenium is used to extract text from the website.
* Document Processing:
  * The extracted text is split into chunks using RecursiveCharacterTextSplitter.
  * The chunks are stored in a vector database using FAISS.
* Chain Creation: A prompt and model chain is created using LangChain and Google's Gemma model.
* Query Handling:
  * The user can ask a query.
  * The query is used to fetch relevant context from the vector database.
  * The context is passed into the prompt.
  * The language model generates an answer.
  * The user can continue to interact with the AI through chat.
* Running the Application: To run the application, use the following command:

```bash
streamlit run your_script.py
```

Replace your_script.py with the name of your Python script.

## Conclusion
This RAG Streamlit application leverages powerful tools for web scraping, document processing, and conversational AI to provide an interactive and intelligent user experience. Users can input a website URL and engage in a meaningful conversation with the AI based on the content extracted from the website.
