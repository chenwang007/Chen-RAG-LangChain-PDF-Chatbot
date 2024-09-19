import streamlit as st
import dotenv
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from tempfile import NamedTemporaryFile

# Load environment variables
dotenv.load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Streamlit app title
st.title("PDF Question Answering with and without RAG")

# File uploader for the PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Function to load PDF
def load_pdf(file):
    if file is not None:
        # Save the uploaded PDF to a temporary file
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        # Load PDF using the temporary file path
        loader = PyPDFLoader(temp_file_path)
        return loader.load()
    return None

# Load the PDF document
document = None
if uploaded_file:
    document = load_pdf(uploaded_file)
    st.write("PDF loaded successfully.")

# Text area for the question with increased size
question = st.text_area("Enter your question:", height=150)

# Add submit button for the question input
if st.button("Submit Question") and question and document:
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(document)
    
    # Create a vector store from the document chunks
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    
    # Set up retriever
    retriever = vectorstore.as_retriever()
    
    # Pull RAG prompt from the hub
    prompt = hub.pull("rlm/rag-prompt")
    
    # Function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # RAG chain setup
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Display the question
    st.write(f"Question: {question}")
    
    # Answer without RAG
    st.subheader("Answer without RAG:")
    answer_without_rag = llm.invoke([question]).content
    st.write(answer_without_rag)
    
    # Answer with RAG
    st.subheader("Answer with RAG:")
    answer_with_rag = rag_chain.invoke(question)
    st.write(answer_with_rag)
    
    # Clean up vector store
    vectorstore.delete_collection()
