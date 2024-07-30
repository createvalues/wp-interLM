from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, \
    StorageContext, load_index_from_storage

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

embed_model = HuggingFaceEmbedding(
    model_name="/root/models/sentence-transformer"
)

Settings.embed_model = embed_model

llm = HuggingFaceLLM(
    model_name="/root/models/internlm2-chat-1_8b",
    tokenizer_name="/root/models/internlm2-chat-1_8b",
    model_kwargs={"trust_remote_code":True},
    tokenizer_kwargs={"trust_remote_code":True}
)
Settings.llm = llm

# documents = SimpleDirectoryReader("/root/llamaindex/data").load_data()
# index = VectorStoreIndex.from_documents(documents)
# # 将embedding向量和向量索引存储到文件中
# index.storage_context.persist(persist_dir='./doc_emb')

# 从存储文件中读取embedding向量和向量索引
storage_context = StorageContext.from_defaults(persist_dir="./doc_emb")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()
response1 = query_engine.query("xtuner是什么?")
print(response1)
print("-"*50)
response2 = query_engine.query("医生你好，我月经量少，这次月经前天来的，昨天就一点点，今天就没了，请问月经量少是什么引起的?")
print(response2)
