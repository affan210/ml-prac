from dotenv import load_dotenv
from typing import Union, List, Tuple
from langchain.schema import AgentAction, AgentFinish

from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from langchain.tools import tool, Tool
from langchain.tools.render import render_text_description_and_args, render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser

from agent_tools.custom_callbacks import CustomAgentCallbackHandler
from agent_tools.output_parser import text_length_parser, FinalAnswerModel

@tool
def get_text_length(text: str, topic: str) -> int:
    """Get the length of a string."""
    # print(f"get_text_length called with text: {text}")
    text = text + topic # concatenating the text and topic
    text = text.strip("\n").strip('"') # stripping non-alphanumeric characters
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    """Find a tool by its name."""
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found.")

def format_log_str(
        intermediate_steps: List[Tuple[AgentAction, str]], 
        observation_prefix: str = "Observation:", 
        llm_prefix: str = "Thought:"
    ) -> str:
    """ Construct the scratchpad that lets the agent continue its thought process """
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix} {observation}\n{llm_prefix}"
    # print(f"format_log_str: {thoughts=}")
    return thoughts

if __name__ == "__main__":
    load_dotenv()
    print("\nHello LangChain ReAct Agent!\n")
    # res = get_text_length.invoke(input={"text": "topic", "topic": "hello world"})
    
    user_input_to_llm = "What is the length of the text: dogg ? I only need the number"

    harrison_chase_modified_prompt = """
        You have access to the following tools:
        {tools}

        Please use the following format:

        Question: the input question you must answer
        Thought: Do I need to use a tool? Yes/No I need/need not to use a tool
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
        {{
        Thought: I know the final answer, Do I need to use a tool? No
        Final Answer: [your response or final answer to the original input Question here]
        }}
        Do not attach backticks or quotes to any of your outputs, including thoughts, tool calls and final answers.

        Begin!

        Question: {input}
        Thought: Do I need to use a tool? {agent_scratchpad}
    """
    # agent_scratchpad : brings back the previous thought process (history) of the agent
    
    llm = ChatOllama(
        model="gemma2:2b",
        temperature=0,
        stop=["\nObservation"],
        callbacks=[CustomAgentCallbackHandler()],
        # prompt_template=prompt_template
    )
    
    tools = [
        get_text_length
    ]
    
    prompt_template = PromptTemplate.from_template(template=harrison_chase_modified_prompt).partial(
        tools=render_text_description( tools), 
        tool_names=", ".join([tool.name for tool in tools])
    )
    
    intermediate_steps = []

    agent = ( # | = pipe  LCEL (Langchain Expression Language) operator, which is a way to chain components together in LangChain
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_str(x["agent_scratchpad"]),
         } 
         | prompt_template 
         | llm 
         | text_length_parser)
    
    print(agent)

    """## # initial implementation of the agent executor by myself # ##
    
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": user_input_to_llm,
            "agent_scratchpad": intermediate_steps
        }
    )
    # print(agent_step)
    print("Agent Executor Steps:")
    print(f"{'-'*30}\n")
    print(agent_step.model_dump().get("log"))

    react_agent_step_counter = 1
    if isinstance(agent_step, AgentAction):
        while not isinstance(agent_step, AgentFinish):
            react_agent_step_counter += 1
            
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input), "")  # 2nd argument is the topic of function get_text_length, passing blank as of now
            # print(f"{observation=}")
            intermediate_steps.append((agent_step, str(observation)))

            # Re-invoke the agent with the updated intermediate steps and observations
            agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
                {
                    "input": user_input_to_llm,
                    "agent_scratchpad": intermediate_steps
                }
            )
            print(agent_step.model_dump().get("log"))
            if react_agent_step_counter > 5:
                print("Too many thought processes, breaking out of the thinking.")
                break
        if isinstance(agent_step, AgentFinish):
            print("-"*30)
            print(agent_step.return_values.get("output"))
        #############################################################################
        """
    
    # easy while loop implementation of the agent executor
    print("Agent Executor Steps:")
    print(f"{'-'*30}\n")
    agent_step = ""
    react_agent_step_counter = 1
    while not isinstance(agent_step, FinalAnswerModel):
        react_agent_step_counter += 1
        # Invoke the agent with the intermediate steps and observations, the loop will reinvoke the agent with the updated intermediate steps and observations
        agent_step: Union[AgentAction, AgentFinish, FinalAnswerModel] = agent.invoke({
                "input": user_input_to_llm,
                "agent_scratchpad": intermediate_steps
            }
        )
        if isinstance(agent_step, AgentAction):
            print(agent_step.model_dump().get("log"))
            
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input), "")  # 2nd argument is the topic of custom function get_text_length which contains the business logic, passing blank as of now
            intermediate_steps.append((agent_step, str(observation)))

        if react_agent_step_counter > 5: # to avoid infinite loop
            print("Too many thought processes, breaking out of the thinking.")
            break
            
    if isinstance(agent_step, FinalAnswerModel):
        print("-"*30)
        print(agent_step.to_dict())
        # print(agent_step.return_values.get("output"))
    
    # above is the same as below 
    # from langchain.agents import initialize_agent, AgentType
    # agent_executor = initialize_agent(
    #     tools=tools,
    #     llm=llm,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    #     agent_kwargs={
    #         "agent_scratchpad": format_log_str(intermediate_steps)
    #     },
    # )