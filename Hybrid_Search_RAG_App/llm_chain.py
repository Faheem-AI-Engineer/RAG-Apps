from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEndpoint

def initialize_llm(repo_id):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation"
    )

def create_retrieval_chain(retriever, llm):
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    def safe_retrieval(question):
        try:
            return format_docs(retriever.invoke(question))
        except Exception:
            return "No relevant context found."

    prompt_template = ChatPromptTemplate.from_template("""
        You are an intelligent assistant designed to provide accurate and concise answers based on the given context. Follow these guidelines strictly:
        1. Use only the information provided in the context to answer the question.
        2. If the context does not contain enough information to answer the question, respond with "I don't know."
        3. Ensure your answer is directly relevant to the question and avoids unnecessary details.
        4. Do not mention that the answer is derived from the context.
        5. Please ensure that your answer is organized and easy to follow. Use headings, bullet points, numbered lists, and clear sections to structure your response.

        ### Question: {question}

        ### Context: {context}

        ### Answer:
    """)

    return (
        {"context": RunnableLambda(safe_retrieval), "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )