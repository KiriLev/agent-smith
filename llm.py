from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
import json

load_dotenv()
client = OpenAI()


class LLMResponse(BaseModel):
    success: bool
    results: Optional[list] = None


def run_llm(system_prompt, user_prompt) -> LLMResponse:
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {"role": "system",
             "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    print(completion.choices[0].message.content)
    response = LLMResponse(**json.loads(completion.choices[0].message.content))
    return response
