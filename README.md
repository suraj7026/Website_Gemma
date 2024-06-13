# RAG Streamlit Application

This project is a Retrieval-Augmented Generation (RAG) application built using Streamlit. It allows users to input a URL of a website, extract text from the website using Selenium, split the text into chunks, store these chunks in a vector database, and create a conversational AI using LangChain with Google's Gemma model. Users can then interact with the AI by asking queries, which fetch context from the vector database and generate answers.

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
