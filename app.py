import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(
    page_title="AI Shopping Assistant",
    layout="wide"
)

st.title("AI Shopping Recommendation Assistant")

st.write(
    "Describe what product you want"
)

df = pd.read_csv(
    "products.csv"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(
    df["description"].tolist()
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    np.array(
        embeddings
    )
)

query = st.text_input(
    "Search"
)

if query:

    query_embedding = model.encode(
        [query]
    )

    distances, indices = index.search(
        np.array(
            query_embedding
        ),
        3
    )

    st.subheader(
        "Recommended Products"
    )

    for idx in indices[0]:

        st.write(
            f"""
            Product: {df.iloc[idx]['name']}
            
            Description: {df.iloc[idx]['description']}
            
            Price: ${df.iloc[idx]['price']}
            """
        )