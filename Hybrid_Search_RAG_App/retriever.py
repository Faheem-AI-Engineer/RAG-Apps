from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from pinecone_text.sparse import BM25Encoder

def initialize_pinecone(api_key, index_name):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)

def create_hybrid_retriever(index, model_name):
    embeddings = FastEmbedEmbeddings(model_name=model_name)
    bm25encoder = BM25Encoder().default()
    return PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25encoder,
        index=index,
        alpha=1
    ), embeddings