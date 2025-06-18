# import importlib
# import llama_index

# importlib.reload(llama_index)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.core.settings import Settings

# --------------------------------------------------------------------------------------------

documents = SimpleDirectoryReader('/content/data').load_data()

# --------------------------------------------------------------------------------------------

prompt = "You are a helpful and knowledgeable assistant that answers questions based on documents."
query_wrapper_prompt = PromptTemplate("User: {query_str}\nAssistant:")

# --------------------------------------------------------------------------------------------

import torch
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="tiiuae/falcon-7b-instruct",
    model_name="tiiuae/falcon-7b-instruct",
    device_map="auto",
    #uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
)

# --------------------------------------------------------------------------------------------

# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from llama_index.core import ServiceContext
# # from llama_index.embeddings.langchain import LangchainEmbedding
# embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Load a sentence-transformers model
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Wrap for LlamaIndex
embed_model = LangchainEmbedding(hf_embeddings)

# --------------------------------------------------------------------------------------------


Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024

# --------------------------------------------------------------------------------------------

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

# --------------------------------------------------------------------------------------------

response = query_engine.query("Who is the FBI authorized to assist with investigations according to this regulation")
print(response)





