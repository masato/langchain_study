import os  # noqa: D100
import sys

import lancedb
import pyarrow as pa
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores.lancedb import LanceDB
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()


def initialize_vectorstore() -> LanceDB:  # noqa: D103
    db = lancedb.connect(os.environ["LANCEDB_DB"])
    try:
        table = db.open_table(os.environ["LANCEDB_TABLE"])
    except FileNotFoundError:
        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), list_size=1536)),
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("source", pa.string()),
            ],
        )
        table = db.create_table(
            os.environ["LANCEDB_TABLE"],
            schema=schema,
        )

    embeddings = OpenAIEmbeddings()

    return LanceDB(
        table,
        embeddings,
    )


if __name__ == "__main__":
    file_path = sys.argv[1]
    loader = UnstructuredMarkdownLoader(file_path)
    raw_docs = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
    )
    docs = text_splitter.split_documents(raw_docs)

    vectorstore = initialize_vectorstore()
    vectorstore.add_documents(docs)
