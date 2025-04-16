import os
from dotenv import load_dotenv
from typing import Tuple, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import linkedin_lookup
from agent_tools.output_parser import summary_parser, Summary

def ice_break_linked_url(name: str) -> Tuple[Summary, str]:
    linkedin_username = linkedin_lookup(name=name)
    linkedin_info = scrape_linkedin_profile(url = linkedin_username)

    query_template = """given the LinkedIn profile information {information} of a person, I want you to create:
     1. a short summary of the person in 2-3 sentences
     2. interesting facts about the person
     Return the result in the following JSON format:
    {{
        "summary": "<A brief summary of the person>",
        "facts": ["<Interesting fact 1>", "<Interesting fact 2>", ...]
    }}
    """
    #  \n{format_instructions}""" # not working with gemma2, hence manual prompt for output formating needed (line 23-27) to comply with pydantic summary_parser object
    
    query_prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=query_template,
        # partial_variables={"format_instructions": summary_parser.get_format_instructions()}, # not working with gemma2
    )

    llm = ChatOllama(temperature=0, model="gemma2:2b")

    chain = query_prompt_template | llm | summary_parser # | StrOutputParser() # this is a chain of components that will be executed in order 

    res = chain.invoke(input={"information": linkedin_info})
    print(res)
    print(res.to_dict())

    return res, linkedin_info.get("photoUrl")

if __name__ == "__main__":
    load_dotenv()
    print("Hello LinkedIn LangChain!")
    # print(f"{os.environ['OPENAI_API_KEY']}")
    print("-"*20)
    ice_break_linked_url(name="Eden Marco")

    # linkedin_info = scrape_linkedin_profile("https://www.linkedin.com/in/eden-marco/")


