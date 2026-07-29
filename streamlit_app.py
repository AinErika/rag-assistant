import streamlit as st
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from string import Template
import asyncio
import os
from dotenv import load_dotenv

# ============================================================
# 1. Импорт ваших классов из проекта
# ============================================================
from src.components.llm import OpenAILikeLLM, InputMessage
from src.settings import settings

# ============================================================
# 2. Ваши классы NGramKnowledgeStorage и NGramRetriever
# ============================================================
class NGramKnowledgeStorage:
    def __init__(self, n: int) -> None:
        self.n = n
        self.documents = {}
        self.inverted_index = defaultdict(set)

    def load_documents_from_file(self, data_path: Path) -> 'NGramKnowledgeStorage':
        with open(data_path, encoding="utf-8") as f:
            documents = f.read().split("\n\n")
        for index, text in enumerate(documents):
            if not text.strip():
                continue
            doc_id = index + 1
            self.documents[doc_id] = text
            ngrams = self.extract_ngrams(text)
            for ngram in ngrams:
                self.inverted_index[ngram].add(doc_id)
        return self

    def extract_ngrams(self, text: str) -> set[str]:
        text = text.lower()
        ngrams = set()
        for i in range(len(text) - self.n + 1):
            ngrams.add(text[i:i + self.n])
        return ngrams

    def get_document_text(self, doc_id: int) -> str:
        return self.documents[doc_id]

    def check_ngram(self, ngram: str) -> bool:
        return ngram in self.inverted_index

    def get_doc_ids_by_ngram(self, ngram: str) -> set[int]:
        return self.inverted_index[ngram]


@dataclass
class RetrieveResult:
    doc_id: int
    score: int
    text: str


class NGramRetriever:
    def __init__(self, storage: NGramKnowledgeStorage) -> None:
        self.storage = storage

    def retrieve(self, query: str, top_k: int) -> list[RetrieveResult]:
        query_ngrams = self.storage.extract_ngrams(query)
        candidate_docs = defaultdict(int)
        for ngram in query_ngrams:
            if self.storage.check_ngram(ngram):
                for doc_id in self.storage.get_doc_ids_by_ngram(ngram):
                    candidate_docs[doc_id] += 1
        sorted_docs = sorted(candidate_docs.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:top_k]
        return [
            RetrieveResult(doc_id, count, self.storage.get_document_text(doc_id))
            for doc_id, count in top_docs
        ]


# ============================================================
# 3. Настройка промптов
# ============================================================
SYSTEM_PROMPT = Template("You are a useful assistant. Use the language specified in $lang for your entire response.")
USER_PROMPT = Template("Question: $query\n\nContext (the ONLY source of truth):\n$context")


# ============================================================
# 4. Функция загрузки RAG-системы (кэшируется!)
# ============================================================
@st.cache_resource
def load_rag_system():
    """Загружает Knowledge Storage, Retriever и LLM."""
    
    data_path = Path("src/datasets/qi_lir_news/ru.txt")
    
    knowledge_store = NGramKnowledgeStorage(n=3).load_documents_from_file(data_path)
    retriever = NGramRetriever(storage=knowledge_store)
    
    llm = OpenAILikeLLM(
        base_url=settings.llm.BASE_URL,
        api_key=settings.llm.API_KEY,
        model=settings.llm.MODEL,
        common_parameters={"temperature": 0},
    )
    
    return retriever, llm


# ============================================================
# 5. Асинхронная функция для генерации ответа
# ============================================================
async def get_rag_answer(query: str, retriever, llm, top_k: int = 3, lang: str = "russian"):
    """Полный RAG-пайплайн: поиск + генерация."""
    
    external_knowledge = retriever.retrieve(query, top_k=top_k)
    
    system_prompt = SYSTEM_PROMPT.substitute({"lang": lang})
    context = "\n\n".join([f"Новость {doc.doc_id}:\n{doc.text}" for doc in external_knowledge])
    user_prompt = USER_PROMPT.substitute({"query": query, "context": context})
    
    messages = [
        InputMessage(role="system", content=system_prompt),
        InputMessage(role="user", content=user_prompt),
    ]
    
    answer = await llm.generate_answer(messages)
    
    return {
        "answer": answer.content,
        "sources": [{"id": doc.doc_id, "score": doc.score} for doc in external_knowledge],
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
    }


# ============================================================
# 6. Streamlit UI
# ============================================================
def main():
    st.set_page_config(page_title="RAG Ассистент — Ксанф", layout="wide")
    st.title("📰 Ассистент по новостям мира Ксанфа")
    st.markdown("Задайте вопрос о вымышленном мире Ксанфа — я найду ответ в синтетических новостях!")

    with st.spinner("Загрузка RAG-системы..."):
        retriever, llm = load_rag_system()
    
    with st.sidebar:
        st.header("⚙️ Настройки")
        top_k = st.slider("Количество новостей для поиска", 1, 10, 3)
        st.caption("Источник данных: синтетические новости о мире Ксанфа")
        st.divider()
        st.caption(f"Модель: {settings.llm.MODEL}")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Источники информации"):
                    for src in msg["sources"]:
                        st.write(f"Новость #{src['id']} (совпадений: {src['score']})")
    
    if prompt := st.chat_input("Что вы хотите узнать о мире Ксанфа?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Ищу ответ в новостях..."):
                try:
                    result = asyncio.run(get_rag_answer(prompt, retriever, llm, top_k=top_k))
                    
                    full_response = result["answer"]
                    st.markdown(full_response)
                    
                    if result["sources"]:
                        with st.expander("📚 Источники информации"):
                            for src in result["sources"]:
                                st.write(f"Новость #{src['id']} (совпадений: {src['score']})")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": result["sources"]
                    })
                    
                except Exception as e:
                    st.error(f"Ошибка при генерации ответа: {e}")


if __name__ == "__main__":
    main()
# trigger redeploy
# rebuild
# rebuild after removing bcc
# rebuild
# trigger rebuild after package=false
