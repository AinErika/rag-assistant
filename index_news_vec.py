from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import SQLiteVec
import sqlite3

data_path = Path("src/datasets/qi_lir_news/ru.txt")
loader = TextLoader(str(data_path), encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"📄 Создано {len(chunks)} чанков")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

db_file = "news_vec.db"
table_name = "news_embeddings"

Path(db_file).unlink(missing_ok=True)

vectorstore = SQLiteVec.from_documents(
    documents=chunks,
    embedding=embeddings,
    table=table_name,
    db_file=db_file,
)

print(f"✅ Индекс создан! База сохранена в {db_file}")

# Подсчёт записей через прямой SQL-запрос
conn = sqlite3.connect(db_file)
cursor = conn.cursor()
cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
count = cursor.fetchone()[0]
conn.close()
print(f"📊 Количество записей: {count}")
