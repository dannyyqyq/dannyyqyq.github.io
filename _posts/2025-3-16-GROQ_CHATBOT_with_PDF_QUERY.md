---
layout: post
title: Chat with PDF alongside with Search Engine using GROQ!
image: "/posts/chatbot_1.png"
tags: [Python, Machine Learning, Data Science, streamlit, LLM, Huggingface, GROQ, LangChain, Search Engine]
github_repo: "[dannyyqyq/[GROQ CHATBOT with PDF QUERY](https://groq-searchengine-and-pdf.streamlit.app/)](https://github.com/dannyyqyq/F/blob/main/README.md)"
---

# Chat with PDF alongside with Search Engine using GROQ! 

For more details, check out the [project repository on GitHub](https://github.com/dannyyqyq/qroq_search_engine_chatbot_pdf/blob/main/README.md).

This Streamlit application is a mini-project I created to explore the exciting world of Generative AI. It allows you to chat with a combination of web search results and uploaded PDF documents. I built it to learn more about LangChain, Large Language Models (LLMs), and how to integrate various AI tools into a practical application.

## 🚀 Web Application
Experience the CHATBOT live!  
[Live Demo: GROQ search engine CHATBOT with PDF QUERY](https://groq-searchengine-and-pdf.streamlit.app/)

## Features

-   **Chat with PDFs:** Upload a PDF document, and the application will process it, index it, and allow you to ask questions based on its content.
-   **Web Search Integration:** The application integrates with Wikipedia, Arxiv, and DuckDuckGo to provide up-to-date information from the web.
-   **LangChain Agents:** Uses LangChain's agents to intelligently route your questions to the appropriate tools (PDF retrieval or web search).
-   **GROQ LLM:** Leverages the GROQ API and Llama3 for conversational responses.
-   **Hugging Face Embeddings:** Utilizes Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` for efficient document embeddings.
-   **Streamlit UI:** Provides an intuitive and user-friendly interface for interacting with the application.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS and Linux
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Set your GROQ API key:**
    * Enter your GROQ API key in the sidebar's "Settings" section.

2.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

3.  **Upload a PDF (optional):**
    * Use the file uploader to upload a PDF document.

4.  **Start chatting:**
    * Enter your questions in the chat input field and press Enter.

## Code Overview

-   `app.py`: Contains the main Streamlit application code.
-   **Dependencies:**
    -   `langchain_groq`: For interacting with the GROQ API.
    -   `langchain_community`: For document loading, text splitting, vector storage, and embeddings.
    -   `langchain.agents`: For creating and managing LangChain agents.
    -   `streamlit`: For the web application interface.
    -   `sentence-transformers`: For Hugging Face embeddings.
    -   `faiss-cpu`: For efficient vector similarity search.

## Key Components

-   **GROQ LLM:**
    -   Initialized with the provided API key.
    -   Used for generating conversational responses.
-   **Hugging Face Embeddings:**
    -   `sentence-transformers/all-MiniLM-L6-v2` model is used for generating document embeddings.
    -   Embeddings are used to create a FAISS vector store.
-   **FAISS Vector Store:**
    -   Stores the document embeddings for efficient retrieval.
    -   Used to find relevant PDF passages based on user queries.
-   **LangChain Agent:**
    -   Initialized with web search tools (Wikipedia, Arxiv, DuckDuckGo) and the GROQ LLM.
    -   Routes user queries to the appropriate tools.
-   **Streamlit UI:**
    -   Provides an interactive chat interface.
    -   Includes a file uploader for PDF documents.
    -   Displays chat history and search results.

## Future Improvements

-   Add more robust error handling.
-   Implement user authentication.
-   Improve the UI/UX.
-   Explore other embedding models.
-   Add more tools to the LangChain agent.

## Challenges
- Initially the embeddings was done using OPENAI embeddings which needed an additional API key. Therefore i switched to huggingface opensource embeddings which might be less powerful and slower but still get the word done.
