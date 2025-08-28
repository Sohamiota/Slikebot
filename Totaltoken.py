import tiktoken
from langchain_community.document_loaders import PyPDFLoader

pdf_path = "Live Streaming Platform User Guide.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

all_text = " ".join([page.page_content for page in pages])

def count_tokens(text: str, model: str = "gpt-3.5-turbo"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

total_tokens = count_tokens(all_text)
print(f"Total tokens in PDF: {total_tokens}")
