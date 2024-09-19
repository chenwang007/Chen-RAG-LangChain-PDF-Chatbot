# Chen-RAG-LangChain-PDF-Chatbot
PDF Question Answering with and without Retrieval-Augmented Generation (RAG)

This project allows you to load a PDF, split it into manageable chunks, and ask questions both with and without Retrieval-Augmented Generation (RAG).

## Requirements

To run the code, you will need to install the following dependencies:

```bash
pip install dotenv langchain langchain-community langchain-chroma openai
```

## Setup

1. **Install Python dependencies**:
   Run the following command to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API key**:
   Create a `.env` file in the root directory of your project and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Run the Code**:
   To run the script, ensure that the Python file is in the same directory as your PDF (in this case, `ikea_light_switch_manual.pdf`), and run:
   
   ```bash
   python app.py
   ```
