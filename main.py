import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from utils import count_tokens
from vector import retriever

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

template = """
You are an expert in answering questions about Live Streaming Platform of Slike

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input("\nAsk your question (q to quit): ").strip()
    if question.lower() == "q":
        break
    
    reviews = retriever.invoke(question)
    full_prompt = template.format(reviews=reviews, question=question)
    
    result = chain.invoke({"reviews": reviews, "question": question})
    answer = result.content

    input_tokens = count_tokens(full_prompt)
    output_tokens = count_tokens(answer)
    total_tokens = input_tokens + output_tokens

    print(answer)
    print(total_tokens)
