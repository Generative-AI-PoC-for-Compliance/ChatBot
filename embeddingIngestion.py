import openai
import pandas as pd

import numpy as np
import pickle
from transformers import GPT2TokenizerFast
from typing import List

df = pd.read_csv('D:/Temp/GDPR1.csv')
df = df.set_index(["title", "heading"])
print(f"{len(df)} rows in data")
print(df.sample(4))

MODEL_NAME = "ada"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"


def get_embedding(text: str, model: str) -> List[float]:
    result = openai.Embedding.create(
        model=model,
        input=text)
    return result["data"][0]["embedding"]


def get_doc_embedding(text: str) -> List[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)


def get_query_embedding(text: str) -> List[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)


def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], List[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_doc_embedding(r.content.replace("\n", " ")) for idx, r in df.iterrows()
    }


def load_embeddings(fname: str) -> dict[tuple[str, str], List[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    return {
        (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

context_embeddings = compute_doc_embeddings(df)
print(context_embeddings)

example_entry = list(context_embeddings.items())[0]
print(f"{example_entry[0]} : {example_entry[1][:4]}... ({len(example_entry[1])} entries)")

