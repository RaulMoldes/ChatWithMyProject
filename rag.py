import streamlit as st
import ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

PDF_PATH = "data/TFM.pdf"

EMBEDDINGS = OllamaEmbeddings(model="llama3", model_kwargs={'n_gpu_layers': 0, 'n_ctx': 10000, 'n_embd': 768, 'n_head': 12, 'n_layer': 12, 'n_positions': 10000, 'vocab_size': 50256})

def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()
    

def split_text(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator= "\n") 
    return splitter.split_documents(text)


def build_vectorstore(splits):
    # 1. Load and split the text
    # 2. Create Ollama embeddings and vector store
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=EMBEDDINGS)
    vectorstore.save_local("./.vectorstore")
    return print("Vectorstore created successfully!")

def get_vectorstore():
    return FAISS.load_local("./.vectorstore", embeddings=EMBEDDINGS, allow_dangerous_deserialization=True)

 # 3. Call Ollama Llama3 model
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# 4. RAG Setup
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question, vectorstore):
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)





if __name__ == "__main__":
    
    data = load_pdf(PDF_PATH)
    splits = split_text(data)
    # build_vectorstore(splits)

    vectorstore = get_vectorstore()
    
    # Ask a question about the project
    st.title("Chat with my Final Project! 🌐")
    st.caption("This app allows you to chat with my Final Project of my Master's Degree using local Llama-3  embeddings!")
    st.text("Project title: Minería de datos aplicada a sistemas de captura de movimiento.")
    st.text("Author: Raúl Moldes Castillo")
    st.text("E-mail: raul.moldes.work@gmail.com")
    st.text("GitHub: RaulMoldes")
    st.text("LinkedIn: raúl-moldes-castillo")
    st.text("\n")
    prompt = st.text_input("Ask any question about the project")

    # Chat with the webpage
    if prompt:
        result = rag_chain(question = prompt, vectorstore=vectorstore)
        st.write(result)

    st.image("imgs/img.jpeg", caption="Final Project")