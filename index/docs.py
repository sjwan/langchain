from langchain_community.document_loaders import UnstructuredWordDocumentLoader


import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.globals import set_debug
from langchain_community.document_loaders import UnstructuredFileLoader


set_debug(True)


from pathlib import Path


script_dir = os.path.dirname(os.path.abspath(__file__))

db_path = os.path.join(script_dir, "db")
    
vectorstore = None

path = Path(db_path)


if path.is_dir():
    
    vectorstore = Chroma(persist_directory=db_path, embedding_function=OllamaEmbeddings(
                                model="nomic-embed-text",
                                base_url="http://10.91.3.116:11434"
                        ), collection_metadata={"hnsw:space": "cosine"})

else:
    use_element = True
    #文档解析
    if (not use_element):
        from langchain_community.document_loaders import DirectoryLoader
        data_dir = os.path.join(script_dir, "data")
        loader = DirectoryLoader(data_dir)
        docs = loader.load()
    else: 
        data_dir = os.path.join(script_dir, "data/a2.docx")
        loader = UnstructuredFileLoader(data_dir, strategy="fast")
        docs = loader.load()
        

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, persist_directory=db_path, embedding=OllamaEmbeddings(
                                model="nomic-embed-text",
                                base_url="http://10.91.3.116:11434"
                        ), collection_metadata={"hnsw:space": "cosine"}
                    )

retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 6, "score_threshold": 0.1})

docs = retriever.get_relevant_documents("项目编号是多少?")

print(docs)

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

handler = StdOutCallbackHandler()

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
)


# response = rag_chain.invoke({"input": "项目编号是多少?"}, {"callbacks":[handler]})
# print(response)

# for chunk in rag_chain.stream("项目编号是多少?"):
#     print(chunk, end="", flush=True)
# vectorstore.delete_collection()

