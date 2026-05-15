import json
import os
from pathlib import Path
from typing import Any, Dict, List
import unicodedata

from more_itertools import batched

from elasticsearch import Elasticsearch, helpers
from sentence_splitter import SentenceSplitter
from sentence_transformers import SentenceTransformer

# Environment variables
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
INPUT_DIR = os.getenv("INPUT_DIR", "/app/input")
INDEX_NAME = os.getenv("INDEX_NAME", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Batching
DOCUMENT_BATCH_SIZE = int(os.getenv("DOCUMENT_BATCH_SIZE", "150"))
INDEX_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "500"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# Simple sentence embeddings model
model = SentenceTransformer(EMBEDDING_MODEL)

# Sentence Splitter
splitter = SentenceSplitter(language="en")


def create_index(es: Elasticsearch, index_name: str) -> None:
    # Create the index if it does not exist.
    if es.indices.exists(index=index_name):
        return

    mapping = {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "description": {"type": "text"},  # required
                "embedding": {
                    "type": "dense_vector",
                    "dims": model.get_sentence_embedding_dimension(),  # each embeddings model has different dimensions
                },
            }
        }
    }

    es.indices.create(index=index_name, body=mapping)
    print(f"Created index: {index_name}")


def load_json_files(input_dir: str) -> List[Dict[str, Any]]:
    documents = []
    for path in Path(input_dir).glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            documents.append(json.load(f))
    return documents


def split_into_chunks(text: str, max_sentences: int = 5) -> List[str]:
    # Split the text into small chunks.
    sentences = splitter.split(text)
    chunks = []

    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i : i + max_sentences]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def generate_embeddings(text: str | List[str]) -> List[float]:
    # Generate embedding for a single text chunk or a list of text chunks
    embeddings = model.encode(
        text, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=False
    )
    return embeddings.tolist()


def to_ascii(text: str) -> str:
    # Replace non-ASCII characters with their closest ASCII equivalent
    # If no ASCII equivalent, discard the character
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def process_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_chunks = []
    all_docs_processed = []

    for document in documents:
        doc_id = document.get("id")
        description = document.get("description", "")

        if not doc_id or not description:
            # id and description are required fields
            raise ValueError("Document must contain at least 'id' and 'description'")

        title = to_ascii(document.get("title", ""))
        description = to_ascii(description)
        authors = [to_ascii(a) for a in document.get("authors", [])]
        subjects = [to_ascii(s).capitalize() for s in document.get("subjects", [])]

        chunks = split_into_chunks(description)
        all_chunks.extend(chunks)

        for idx, chunk in enumerate(chunks):
            all_docs_processed.append(
                {
                    "doc_id": str(doc_id),
                    "chunk_id": f"{doc_id}-{idx}",
                    "title": title,
                    "description": chunk,
                    # "embedding": generate_embedding(chunk),
                    "authors": authors,
                    "first_publish_year": document.get("first_publish_year", ""),
                    "subjects": subjects,
                    "language": document.get("language", []),
                    "openlibrary_url": document.get("openlibrary_url", ""),
                }
            )
    embeddings = generate_embeddings(all_chunks)
    for doc_processed, embedding in zip(all_docs_processed, embeddings):
        doc_processed["embedding"] = embedding

    return all_docs_processed


def index_documents(
    es: Elasticsearch, index_name: str, docs: List[Dict[str, Any]]
) -> None:
    if not docs:
        return

    actions = [
        {
            "_op_type": "index",
            "_index": index_name,
            "_id": doc["chunk_id"],
            "_source": doc,
        }
        for doc in docs
    ]

    # Bulk indexing since we are already batching the documents
    _, errors = helpers.bulk(
        es, actions, chunk_size=INDEX_BATCH_SIZE, raise_on_error=False
    )

    if errors:
        raise ValueError(f"Bulk indexing errors: {errors}")
    print(f"Indexed {len(docs)} chunks")


def semantic_search(
    es: Elasticsearch, index_name: str, query_text: str, k: int = 3
) -> Dict[str, Any]:
    # Query to perform semantic search
    query_vector = generate_embeddings(query_text)

    body = {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": 10,
        },
        "_source": ["doc_id", "chunk_id", "title", "description"],
    }

    return es.search(index=index_name, body=body)


def print_results(query: str, results: Dict[str, Any]) -> None:
    print()
    print("=" * 80)
    print(f"Query: {query}")
    hits = results.get("hits", {}).get("hits", [])
    if not hits:
        print("    No results")
        return
    for rank, hit in enumerate(hits, start=1):
        src = hit.get("_source", {})
        print(
            f"  {rank}. title={src.get('title')!r} doc={src.get('doc_id')} chunk={src.get('chunk_id')}"
        )
        print(f"     {src.get('description')!r}")


def main() -> None:
    es = Elasticsearch(ELASTICSEARCH_URL)
    create_index(es, INDEX_NAME)
    documents = load_json_files(INPUT_DIR)

    if not documents:
        print("No JSON files found.")
        return

    for document_batch in batched(documents, DOCUMENT_BATCH_SIZE):
        all_docs_processed = process_documents(document_batch)
        index_documents(es, INDEX_NAME, all_docs_processed)

    print("Semantic search: examples")

    example_queries = [
        "Crime/detective novel about a serial killer",
        "Drama novel set in the victorian era",
        "Dystopian novel with a political and/or ethical critique",
        "Horror novel about a haunted house",
    ]
    for query in example_queries:
        results = semantic_search(es, INDEX_NAME, query, k=3)
        print_results(query, results)

def run_semantic_searches(k: int = 3) -> None:
    es = Elasticsearch(ELASTICSEARCH_URL)
    
    example_queries = [
        "Crime/detective novel about a serial killer",
        "Drama novel set in the victorian era",
        "Dystopian novel with a political and/or ethical critique",
        "Horror novel about a haunted house",
    ]
    for query in example_queries:
        results = semantic_search(es, INDEX_NAME, query, k=3)
        print_results(query, results)


if __name__ == "__main__":
    main()
    # run_semantic_searches()
