import streamlit as st
from pathlib import Path
from string import Template
import asyncio
from dotenv import load_dotenv

# Импорты из вашего проекта
from src.components.llm import OpenAILikeLLM, InputMessage
from src.settings import settings

# Импортируем наш новый ретривер
from retriever_chroma import ChromaRetriever

# Загружаем переменные окружения
load_dotenv()

# Промпты (как были)
SYSTEM_PROMPT = Template("You are a useful assistant. Use the language specified in $lang for your entire response.")
USER_PROMPT = Template("Question: $query\n\nContext (the ONLY source of truth):\n$context")

@st.cache_resource
def load_rag_system():
    """Загружает ретривер и LLM (кэшируется)"""
    retriever = ChromaRetriever(persist_directory="chroma_db", k=5)
    llm = OpenAILikeLLM(
        base_url=settings.llm.BASE_URL,
        api_key=settings.llm.API_KEY,
        model=settings.llm.MODEL,
        common_parameters={"temperature": 0},
    )
    return retriever, llm

async def get_rag_answer(query: str, retriever, llm, top_k: int = 3, lang: str = "russian"):
    """Полный RAG-пайплайн: поиск + генерация"""
    # 1. Поиск
    docs = retriever.retrieve(query, top_k=top_k)
    
    # 2. Формирование контекста
    context = "\n\n".join([f"Источник {i+1}:\n{doc['text']}" for i, doc in enumerate(docs)])
    
    # 3. Промпты
    system_prompt = SYSTEM_PROMPT.substitute({"lang": lang})
    user_prompt = USER_PROMPT.substitute({"query": query, "context": context})
    messages = [
        InputMessage(role="system", content=system_prompt),
        InputMessage(role="user", content=user_prompt),
    ]
    
    # 4. Генерация ответа
    answer = await llm.generate_answer(messages)
    
    return {
        "answer": answer.content,
        "sources": [{"id": i+1, "text": doc["text"][:100] + "..."} for i, doc in enumerate(docs)],
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
    }

def main():
    st.set_page_config(page_title="RAG Ассистент — Ксанф (LangChain + Chroma)", layout="wide")
    st.title("📰 Ассистент по новостям мира Ксанфа (векторный поиск)")
    st.markdown("Задайте вопрос — я найду ответ в новостях, используя семантический поиск.")

    with st.spinner("Загрузка RAG-системы..."):
        retriever, llm = load_rag_system()

    with st.sidebar:
        st.header("⚙️ Настройки")
        top_k = st.slider("Количество фрагментов для поиска", 1, 10, 3)
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
                        st.write(f"Источник #{src['id']}: {src['text']}")

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
                                st.write(f"Источник #{src['id']}: {src['text']}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": result["sources"]
                    })
                except Exception as e:
                    st.error(f"Ошибка при генерации ответа: {e}")

if __name__ == "__main__":
    main()
