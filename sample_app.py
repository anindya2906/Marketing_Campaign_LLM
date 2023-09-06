from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
    LengthBasedExampleSelector,
)

load_dotenv()

examples = [
    {
        "user_query": "Hi, can you please tell me the time.",
        "bot_answer": "Its time to buy a new watch.",
    },
    {
        "user_query": "Can you suggest me some movies to watch.",
        "bot_answer": "Sure, I thought you life was a big movie enough",
    },
]

example_template = """
user: {user_query}
bot: {bot_answer}
"""

example_prompt = PromptTemplate(
    template=example_template, input_variables=["user_query", "bot_answer"]
)

prefix = """You are a funny chatbot which responds very sarcastically to user queries.
Here are some examples:
"""
suffix = """user: {user_query}
bot: """


example_selector = LengthBasedExampleSelector(
    examples=examples, example_prompt=example_prompt, max_length=100
)


prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    example_separator="\n\n",
    prefix=prefix,
    suffix=suffix,
    input_variables=["user_query"],
)

# print(prompt.format(user_query="What should I eat?"))

llm = OpenAI()

while True:
    res = llm(prompt.format(user_query=input("Ask: ")))
    print("Answer: ", res)
