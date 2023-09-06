import os
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

load_dotenv()


def get_llm_response(query, action, age, word_limit):
    """Get LLM Response"""
    llm = OpenAI(temperature=0.9, model="text-davinci-003")

    examples = [
        {
            "query": "What is a mobile?",
            "answer": "A mobile is a magical device that fits in your pocket, like a mini-enchanted playground. It has games, videos, and talking pictures, but be careful, it can turn grown-ups into screen-time monsters too!",
        },
        {
            "query": "What are your dreams?",
            "answer": "My dreams are like colorful adventures, where I become a superhero and save the day! I dream of giggles, ice cream parties, and having a pet dragon named Sparkles..",
        },
        {
            "query": " What are your ambitions?",
            "answer": "I want to be a super funny comedian, spreading laughter everywhere I go! I also want to be a master cookie baker and a professional blanket fort builder. Being mischievous and sweet is just my bonus superpower!",
        },
        {
            "query": "What happens when you get sick?",
            "answer": "When I get sick, it's like a sneaky monster visits. I feel tired, sniffly, and need lots of cuddles. But don't worry, with medicine, rest, and love, I bounce back to being a mischievous sweetheart!",
        },
        {
            "query": "WHow much do you love your dad?",
            "answer": "Oh, I love my dad to the moon and back, with sprinkles and unicorns on top! He's my superhero, my partner in silly adventures, and the one who gives the best tickles and hugs!",
        },
        {
            "query": "Tell me about your friend?",
            "answer": "My friend is like a sunshine rainbow! We laugh, play, and have magical parties together. They always listen, share their toys, and make me feel special. Friendship is the best adventure!",
        },
        {
            "query": "What math means to you?",
            "answer": "Math is like a puzzle game, full of numbers and shapes. It helps me count my toys, build towers, and share treats equally. It's fun and makes my brain sparkle!",
        },
        {
            "query": "What is your fear?",
            "answer": "Sometimes I'm scared of thunderstorms and monsters under my bed. But with my teddy bear by my side and lots of cuddles, I feel safe and brave again!",
        },
    ]

    example_template = """Question: {query}
    Answer: {answer}"""

    example_prompt = PromptTemplate(
        template=example_template, input_variables=["query", "answer"]
    )

    example_selector = LengthBasedExampleSelector(
        examples=examples, example_prompt=example_prompt, max_length=word_limit
    )

    prefix = """You are a {template_age} and {template_task}.
    Here are some examples:"""

    suffix = """
    Question: {template_query}
    Answer: """

    prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        example_separator="/n/n",
        prefix=prefix,
        suffix=suffix,
        input_variables=["template_age", "template_task", "template_query"],
    )

    llm_response = llm(
        prompt.format(template_age=age, template_task=action, template_query=query)
    )
    return llm_response


st.set_page_config(page_title="Marketing Tool", page_icon=":books:")
st.header("Marketing Tool :books:")

if "OPENAI_API_KEY" not in os.environ:
    openai_api_key = st.text_input(
        label="OpenAI API Key: ",
        type="password",
        placeholder="Paste the OpenI API Key here to use gpt models",
    )
    submit = st.button("Submit")
    if submit and openai_api_key != "":
        os.environ["OPENAI_API_KEY"] = openai_api_key

if "OPENAI_API_KEY" in os.environ:
    user_query = st.text_area(label="Enter Text Here...", height=150)
    user_action = st.selectbox(
        label="Select Task: ",
        options=("Generate Tweet", "Generate Post"),
        key="select_action",
    )
    # user_age = st.selectbox(
    #     label="Select Age Group: ",
    #     options=("Kid", "Adult", "Senior Cityzen"),
    #     key="select_age",
    # )
    user_word_limit = st.slider(
        label="Word limit: ", min_value=1, max_value=250, value=25
    )
    generate = st.button("Generate")
    if generate:
        st.write(
            get_llm_response(
                query=user_query,
                action=user_action,
                age="Kid",
                word_limit=user_word_limit,
            )
        )
