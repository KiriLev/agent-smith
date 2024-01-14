import os

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

load_dotenv()

openai_client = OpenAI()

together_client = OpenAI(base_url="https://api.together.xyz/v1",
                         api_key=os.environ["TOGETHER_AI_API_KEY"])


class LLMResponse(BaseModel):
    success: bool
    results: Optional[list] = None


def run_mistral_llm(system_prompt, user_prompt) -> LLMResponse:
    print("running mistral on Together.ai")
    completion = together_client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.2,
        messages=[
            {"role": "system",
             "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    content = completion.choices[0].message.content
    # together.ai output doesn't enforce json output
    content = content[content.find("{"): content.find("}")+1]
    print("Together.ai mistral output:", content)
    response = LLMResponse.model_validate_json(content)
    return response


def run_mistral_tuned_llm(system_prompt, user_prompt) -> LLMResponse:
    print("running mistral_tuned on Together.ai")
    completion = together_client.chat.completions.create(
        model="zetyquickly@googlemail.com/Mistral-7B-Instruct-v0.2-finetune-example-2024-01-14-03-07-26",
        temperature=0.2,
        messages=[
            {"role": "system",
             "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    content = completion.choices[0].message.content
    # together.ai output doesn't enforce json output
    content = content[content.find("{"): content.find("}")+1]
    print("Together.ai mistral tuned output:", content)
    response = LLMResponse.model_validate_json(content)
    return response


def run_openai_llm(system_prompt, user_prompt) -> LLMResponse:
    print("running OpenAI model")
    completion = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {"role": "system",
             "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    content = completion.choices[0].message.content
    print("OpenAI output:", content)
    # breakpoint()
    # breakpoint()
    response = LLMResponse.model_validate_json(content)
    return response

