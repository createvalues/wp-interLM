from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, \
    StorageContext, load_index_from_storage

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

embed_model = HuggingFaceEmbedding(
    model_name="/root/models/sentence-transformer"
)

Settings.embed_model = embed_model

# llm = HuggingFaceLLM(
#     model_name="/root//wp-interLM/lora/XTuner/merged",
#     tokenizer_name="/root//wp-interLM/lora/XTuner/merged",
#     model_kwargs={"trust_remote_code":True},
#     tokenizer_kwargs={"trust_remote_code":True}
# )
llm = HuggingFaceLLM(
    model_name="/root/models/internlm2-chat-1_8b",
    tokenizer_name="/root/models/internlm2-chat-1_8b",
    model_kwargs={"trust_remote_code":True},
    tokenizer_kwargs={"trust_remote_code":True}
)
Settings.llm = llm

print("开始加载知识，，")
documents = SimpleDirectoryReader("/root/llamaindex/data/CMDD_txt").load_data()
print("加载完毕，开始构建索引，，")
index = VectorStoreIndex.from_documents(documents, show_progress=True)
# 将embedding向量和向量索引存储到文件中
print("索引构建完毕，开始保存")
index.storage_context.persist(persist_dir='./doc_emb')
print("知识向量保存至./doc_emb文件夹！")

# # 从存储文件中读取embedding向量和向量索引
# storage_context = StorageContext.from_defaults(persist_dir="./doc_emb")
# index = load_index_from_storage(storage_context)
# query_engine = index.as_query_engine()
response1 = query_engine.query("华妃，你重视皇上吗?")
print(response1)
print("-"*50)
response2 = query_engine.query("华妃，我月经量少，这次月经前天来的，昨天就一点点，今天就没了，请问月经量少是什么引起的?")
print(response2)
