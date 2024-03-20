from dotenv import load_dotenv
from langchain_community.chat_models import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from fastapi import FastAPI
import uvicorn

#fastapi app
app = FastAPI()

#load keys
load_dotenv(".env")

#llm(chatmodel)
llm = ChatCohere()

#embeddings
embeddings = CohereEmbeddings()

@app.get("/input")
def access_chatbot(question):

#prompt
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

#pass document by scraping
    loader = WebBaseLoader("https://allaboutstevejobs.com/blog/")
    docs = loader.load()

#build index
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = faiss.FAISS.from_documents(documents, embeddings)

#chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    result = retrieval_chain.invoke({"input":question})

    return result

if __name__ == '__main__':
    uvicorn(app, host = '127.0.0.1', port = 8000)