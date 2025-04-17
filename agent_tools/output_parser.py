from typing import Dict, Any, List, Optional, Union
from langchain.output_parsers import PydanticOutputParser

# from langchain.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field

from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish

class Summary(BaseModel):
    summary: str = Field(description="summary")
    facts: List[str] = Field(description="facts")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Summary object to a dictionary.
        """
        return {
            "summary": self.summary,
            "facts": self.facts
        }

class FinalAnswerModel(BaseModel):
    final_answer: str
    log: str

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Summary object to a dictionary.
        """
        return {
            "final_answer": self.final_answer,
            "final_response": self.log
        }

class PydanticReActParser(ReActSingleInputOutputParser):
    def parse(self, text: str) -> Union[FinalAnswerModel, AgentAction, AgentFinish]:
        # Call the parent class's parse method
        result = super().parse(text)

        # If the result is an AgentFinish, extract the final answer
        if isinstance(result, AgentFinish):
            final_answer = result.return_values.get("output", "null")
            return FinalAnswerModel(final_answer=final_answer, log=result.log)

        # Otherwise, return the original result (e.g., AgentAction)
        return result
    
summary_parser = PydanticOutputParser(pydantic_object=Summary)
text_length_parser = PydanticReActParser()
# print(summary_parser.get_format_instructions())
# print(summary_parser.get_output_jsonschema())