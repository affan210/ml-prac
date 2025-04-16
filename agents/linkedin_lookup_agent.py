import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Adds the parent directory to the Python path

from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
from agent_tools.tools import get_profile_url_tavily

def linkedin_lookup(name: str) -> str:
    """
    Function to look up a LinkedIn profile based on the provided name.
    """

    llm = ChatOllama(
        model="gemma2:2b", 
        temperature=0
    )
    template = """Given the full name of a person {name_of_person}, return the URL of their LinkedIn profile page.
    Your answer should contain only a URL.
    """
    prompt_template = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )
    tools_for_agent = [
        Tool(
            name="Crawls google for LinkedIn profile page URL",
            func=get_profile_url_tavily,
            description="A tool to look up LinkedIn profiles based on names.",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm=llm, 
        tools=tools_for_agent,
        prompt=react_prompt,
    )
    agent_exec = AgentExecutor(
        agent=agent,
        tools=tools_for_agent,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    result = agent_exec.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)},
    )
    # print(result)
    linkedin_url = result["output"]

    return linkedin_url

if __name__ == "__main__":
    name = "Eden Marco"
    linkedin_url = linkedin_lookup(name)
    print(f"LinkedIn URL for {name}: {linkedin_url}")