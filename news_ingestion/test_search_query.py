import argparse

import chromadb
from sentence_transformers import SentenceTransformer

PERSIST_DIR = './vector-data'


def load_persisted_collection(
    collection_name: str = 'pdf_chunks',
    persist_dir: str = PERSIST_DIR,
):
    """
    Load a ChromaDB collection from the given persistent directory.

    Args:
        collection_name (str, optional): Name of the collection to load.
        persist_dir (str, optional): Path to directory where Chroma persists data.

    Returns:
        chromadb.api.models.Collection.Collection: Loaded ChromaDB collection.
    """
    persistent_client = chromadb.PersistentClient(path=persist_dir)
    collection = persistent_client.get_collection(collection_name)
    print(f"Number of documents in the collection: {collection.count()}")
    return collection

if __name__ == '__main__':  # test_search_query.py
    parser = argparse.ArgumentParser(
        description="Query a persisted ChromaDB collection"
    )
    parser.add_argument(
        "--persist-dir",
        default=PERSIST_DIR,
        help="Path to Chroma persistent directory (default: %(default)s)",
    )
    parser.add_argument(
        "--collection",
        default="pdf_chunks",
        help="Collection name to query (default: %(default)s)",
    )
    parser.add_argument(
        "--query",
        default="Ukraine and Russia Sunflower trade",
        help="Natural language query text (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to return (default: %(default)s)",
    )
    args = parser.parse_args()

    # Load the persisted collection
    collection = load_persisted_collection(
        collection_name=args.collection, persist_dir=args.persist_dir
    )
    print(f"Loaded collection: {collection.name} (persist_dir={args.persist_dir})")

    # Use the SAME SentenceTransformer model as during ingestions
    model = SentenceTransformer('intfloat/multilingual-e5-base')

    # Perform semantic search over documents present in the Chroma collection
    print(f"Performing semantic search for: {args.query}")
    query_embedding = model.encode(args.query)

    # Retrieve top-k most similar documents from the collection
    results = collection.query(
        query_embeddings=[query_embedding],  # list required by chromadb API
        n_results=max(1, args.top_k),
    )

    # Access the retrieved documents and their metadata
    if not results or not results.get('documents'):
        print("No results returned by the collection.")
        raise SystemExit(0)
    documents = results['documents'][0]  # type: ignore[index]
    metadatas = results['metadatas'][0]  # type: ignore[index]
    scores_or_distances = results.get('distances', results.get('scores', [[]]))[0]  # type: ignore[index]
    for i in range(len(documents)):
        doc = documents[i]
        meta = metadatas[i]
        score = scores_or_distances[i]
        # Chroma: smaller distances are closer matches; some configs return
        # "scores" as cosine similarity. If distances are returned, convert to
        # a crude similarity by 1 - distance (cosine metric assumption).
        similarity = None
        if 'scores' in results:  # cosine similarity (1==identical)
            similarity = score
        elif 'distances' in results:  # distance; 0 is perfect
            # crude conversion for cosine; may differ for other metrics
            similarity = 1 - score

        # consider only >80% similarity (adjust threshold as needed)
        if similarity is not None and similarity > 0.1:
            print(f"Result {i+1}:")
            print(doc)
            print(f"Metadata: {meta}")
            print(f"Similarity score: {similarity:.2f}")
            print("---")
