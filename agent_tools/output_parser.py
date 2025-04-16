from typing import Dict, Any, List
from langchain.output_parsers import PydanticOutputParser

# from langchain.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field

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

summary_parser = PydanticOutputParser(pydantic_object=Summary)
# print(summary_parser.get_format_instructions())
# print(summary_parser.get_output_jsonschema())