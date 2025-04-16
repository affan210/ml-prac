import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM

topic = "Agentic AI using langgraph"

if __name__ == "__main__":
    load_dotenv()
    print("Hello LangChain!")
    print(f"{os.environ['OPENAI_API_KEY']}")
    print("-"*20)

    query_template = """given the topic of a technical term, provide a brief definition of the term and an
             example of its. The topic is: {topic}"""
    
    query_prompt_template = PromptTemplate(input_variables=["topic"], template=query_template)

    llm = ChatOllama(temperature=0, model="gemma2:2b")

    chain = query_prompt_template | llm | StrOutputParser() # this is a chain of components that will be executed in order 

    res = chain.invoke(input={"topic": topic})

    print(res)
