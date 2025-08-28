import tiktoken
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about Live Streaming Platform of Slike

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def count_tokens(text: str, model: str = "gpt-3.5-turbo"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

while True:
    question = input("\nAsk your question (q to quit): ")
    if question.lower() == "q":
        break

    reviews = retriever.invoke(question)

    full_prompt = template.format(reviews=reviews, question=question)
    result = chain.invoke({"reviews": reviews, "question": question})

    input_tokens = count_tokens(full_prompt)
    output_tokens = count_tokens(str(result))
    total_tokens = input_tokens + output_tokens

    print(f"\nAnswer: {result}")
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Total tokens: {total_tokens}")
