from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from models import Models

def query_rag(query: str):
    # Initialize models
    models = Models()
    embeddings = models.embeddings_hf
    llm = models.chat_model

    # Initialize Vector DB
    vector_store = Chroma(
        collection_name="documents",
        embedding_function=embeddings,#embedding model
        persist_directory="./db/chroma.db",
    )

    # Define chat prompt
    # needs context as the retrieved information that best matches the query and the actual question that was asked
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer the question based\
            only the data provided."),
            ("human", "Use the user question {input} to answer the question. Use \
            only the {context} to answer the question")
        ]
    )

    # Retrieval chain
    retriever = vector_store.as_retriever(kwargs={"k": 10})
    combine_docs_chain = create_stuff_documents_chain(
        llm, prompt
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    response = retrieval_chain.invoke({"input": query})

    return response["answer"]

def main():
    while True:
        query = input("User (or type 'q', or 'exit' to end): ")
        if query.lower() in ['q', 'exit']:
            break

        #result = retrieval_chain.invoke({"input": query})
        result = query_rag(query)
        print("Assistant: ", result, "\n\n")

if __name__ == "__main__":
    main()
