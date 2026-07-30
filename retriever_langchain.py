import sqlite3
import sqlite_vec
from pathlib import Path
from langchain_community.vectorstores import SQLiteVec
from langchain_community.embeddings import HuggingFaceEmbeddings

class LangChainRetriever:
    def __init__(self, db_file: str = "news_vec.db", table: str = "news_embeddings", k: int = 5):
        self.k = k
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.connection = sqlite3.connect(db_file)
        self.connection.enable_load_extension(True)
        sqlite_vec.load(self.connection)
        self.connection.enable_load_extension(False)
        self.vectorstore = SQLiteVec(
            table=table,
            connection=self.connection,
            embedding=self.embeddings,
        )
    
    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        if top_k is None:
            top_k = self.k
        docs = self.vectorstore.similarity_search(query, k=top_k)
        results = []
        for doc in docs:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": None
            })
        return results

if __name__ == "__main__":
    retriever = LangChainRetriever()
    query = "Как называется столица Ксанфа?"
    results = retriever.retrieve(query, top_k=3)
    print(f"Найдено {len(results)} результатов:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['text'][:150]}...")
