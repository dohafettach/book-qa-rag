"""
Query pipeline.

Takes a question and a book_id, retrieves the most relevant chunks
from Pinecone, and asks GPT to answer using only those chunks.

Flow:
  question → embed → Pinecone similarity search (filtered by book_id)
           → top-K chunks → prompt → GPT-4o-mini → answer + sources
"""

from src.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
)


PROMPT_TEMPLATE = """\
You are a literary assistant that answers questions about books.
Use ONLY the context passages provided below to answer.
If the answer is not in the context, say exactly: "I couldn't find that in the provided passages."
Do not use any outside knowledge.

Context:
{context}

Question: {question}

Answer:\
"""


def ask(question: str, book_id: str) -> dict:
    """
    Ask a question about a specific ingested book.

    Args:
        question: natural language question, e.g. "Why was Dantès imprisoned?"
        book_id:  the slug used when the book was ingested, e.g. "count-of-monte-cristo"

    Returns:
        {
            question:  the original question,
            answer:    GPT's answer grounded in retrieved chunks,
            book_id:   book slug,
            title:     book title from registry,
            sources:   list of { chunk_index, snippet } for each retrieved chunk,
        }

    Raises:
        ValueError if the book_id has never been ingested.
    """
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_pinecone import PineconeVectorStore
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from src.books import get_book

    # Verify this book was actually ingested — fail clearly if not
    book_meta = get_book(book_id)
    if not book_meta:
        raise ValueError(
            f"Book '{book_id}' has not been ingested yet. "
            f"POST to /ingest first."
        )

   
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

   
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": TOP_K,
            "filter": {"book_id": {"$eq": book_id}},
        }
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    result = qa_chain.invoke({"query": question})

    sources = [
        {
            "chunk_index": doc.metadata.get("chunk_index", "?"),
            "snippet":     doc.page_content[:300],
        }
        for doc in result.get("source_documents", [])
    ]

    return {
        "question": question,
        "answer":   result["result"],
        "book_id":  book_id,
        "title":    book_meta["title"],
        "author":   book_meta["author"],
        "sources":  sources,
    }
