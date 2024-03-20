from dotenv import load_dotenv
from langchain_community.chat_models import ChatCohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
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

    chat_history = [HumanMessage(content="Did Steve Jobs work with Wozniak?"), AIMessage(content="Yes!")]
    context =  "Answer the user's questions based on the below context:"

#prompt

    prompt = ChatPromptTemplate.from_messages([
        ("system", "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

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
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    result = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": question,
        "context": context
    })
    return result

if __name__ == '__main__':
    uvicorn(app, host = '127.0.0.1', port = 8000)
