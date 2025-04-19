import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader

from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

# from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    """Format the documents for the prompt."""
    return "\n\n".join([doc.page_content for doc in docs])

if __name__ == "__main__":
    load_dotenv()

    user_query = "What is Pinecone and vecotr db and why is it necessary how is it useful in ml?"

    loader = TextLoader("database/medium-blog-01.txt", encoding="utf-8")
    documents = loader.load()
    # print(documents[0].model_dump().keys())

    print("splitting...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    print(f"{len(texts)} chunks")
    # [print(i) for i in texts]

    embeddings = OllamaEmbeddings(
        model="gemma2:2b", # embedding dimension = 2304
        temperature=0.0,
        top_k=10,
        top_p=0.5,
        tfs_z=1.0,
        repeat_penalty=1.5,
        stop=["\n\n"],
    )
    # # embedding = embeddings.embed_documents(texts) # creates `len(texts)` same number of embeddings as chunks
    # # [print(len(i)) for i in embedding] # prints the dimension of each embedding | dimension varies with each model

    # # print(embeddings.embed_documents(texts))

    # pc_vs = PineconeVectorStore.from_documents(
    #     documents=texts,
    #     embedding=embeddings,
    #     index_name=os.environ["PINECONE_INDEX_NAME"]
    # )
    # print(pc_vs)

    # query_template = PromptTemplate.from_template(
    #     template=user_query,
    #     # input_variables=["topic"],
    # )

    # llm = ChatOllama(
    #     model="gemma2:2b",
    #     temperature=0.0,
    #     top_k=10,
    #     top_p=0.5,
    #     tfs_z=1.0,
    #     repeat_penalty=1.5,
    #     # stop=["\n\n"],
    # )

    # chain = query_template | llm

    # res = chain.invoke(input={})
    # print(res.content)  
    
    """   
    # ## Assusming the response is not accurate for the asked user_query, we would want to fetch the information from our medium-blog-01 for information
    """

    vector_store = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=embeddings,
    )
    llm = ChatOllama(
        model="gemma2:2b",
        temperature=0.0
    )
    retrival_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    custom_rag_prompt_template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three to four sentences maximum and keep the answer as concise as possible.
    Always say "Happy to help!" at the end of the answer.
    {context}

    Question: {question}
    Helpful Answer:
    """
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=retrival_qa_prompt
    )
    retrival_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )
    # res = retrival_chain.invoke(input={"input": user_query})
    # # print(dir(res))
    # print(res.get("answer"))

    # ## LCEL Implementation
    custom_rag_prompt = PromptTemplate(template=custom_rag_prompt_template)
    rag_chain = (
        {"context": vector_store.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke( user_query)
    print(res.content)