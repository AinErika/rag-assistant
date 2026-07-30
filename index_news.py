import sqlite3
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import SQLiteVSS

# 1. Загрузка документов
data_path = Path("src/datasets/qi_lir_news/ru.txt")
loader = TextLoader(str(data_path), encoding="utf-8")
documents = loader.load()

# 2. Разбиение на чанки
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(documents)

print(f"📄 Создано {len(chunks)} чанков")

# 3. Создание эмбеддингов (бесплатная мультиязычная модель)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 4. Сохранение в SQLiteVSS
db_file = "news_vss.db"
table_name = "news_embeddings"

# Удаляем старую БД, если есть
Path(db_file).unlink(missing_ok=True)

vectorstore = SQLiteVSS.from_documents(
    documents=chunks,
    embedding=embeddings,
    table=table_name,
    db_file=db_file,
)

print(f"✅ Индекс создан! База сохранена в {db_file}")
print(f"📊 Количество записей: {vectorstore._collection.count()}")
