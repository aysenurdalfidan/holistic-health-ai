import streamlit as st
import os
import openai
import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from Bio import Entrez

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

# PubMed Email (required by Entrez)
Entrez.email = "aysenurtskn2@gmail.com"

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Categorized documents
categorized_documents = {
    "Nutrition": [
        "Drinking warm lemon water in the morning helps detoxify your liver and improves digestion.",
        "Intermittent fasting can boost mitochondrial function and support cellular repair.",
        "Eating fermented foods like kimchi and yogurt supports gut microbiome health.",
        "A diet rich in omega-3 fatty acids supports heart and brain health.",
        "Omega-3 fatty acids and B vitamin supplements may be beneficial in the treatment of depression.",
        "Magnesium deficiency can lead to muscle cramps and sleep disturbances.",
        "A low glycemic index diet and regular physical activity are essential for managing insulin resistance.",
        "Personalized nutrition plans and improving insulin sensitivity are crucial for obesity management."
    ],
    "Herbal Remedies": [
        "Chamomile tea is known for its calming effects and can help with sleep and anxiety reduction.",
        "Turmeric has powerful anti-inflammatory properties and supports joint health.",
        "Drinking green tea can enhance brain function and metabolic rate.",
        "Seed cycling can help regulate estrogen and progesterone balance in women.",
        "Magnesium and riboflavin (vitamin B2) supplements are effective in migraine treatment.",
        "Ginkgo biloba supports cognitive function by improving blood circulation.",
        "Ginger may help alleviate nausea.",
        "Milk thistle and artichoke supplements can be beneficial for liver detoxification."
    ],
    "Sleep & Circadian Rhythm": [
        "Getting enough sleep is crucial for hormone regulation and cognitive function.",
        "Blue light exposure at night disrupts melatonin production and sleep quality.",
        "Exposing yourself to morning sunlight helps regulate your circadian rhythm and improves mood.",
        "Practicing good sleep hygiene, such as keeping a consistent sleep schedule, can improve sleep quality.",
        "Magnesium and L-theanine supplements may promote relaxation and support deeper sleep."
    ],
    "Mental Health": [
        "Practicing gratitude daily can reduce stress and improve overall mental health.",
        "Journaling can improve mental clarity and emotional well-being.",
        "Daily meditation helps manage stress and improve concentration levels.",
        "Spending time in nature can reduce anxiety and promote overall well-being.",
        "Balanced blood sugar levels can contribute to better mood stability and reduced anxiety."
    ],
    "Detox": [
        "Sweating through exercise or sauna supports detoxification by eliminating heavy metals.",
        "Cold exposure can stimulate mitochondrial biogenesis, improving energy production.",
        "Imbalances in the gut microbiome are linked to autoimmune diseases, and this balance can be restored with probiotic and prebiotic support.",
        "Identifying the root causes of chronic inflammation and treating it through an anti-inflammatory diet and lifestyle changes is possible.",
        "Activated charcoal may help with toxin removal from the digestive system."
    ],
    "Hormone Health": [
        "Estrogen dominance can lead to symptoms such as mood swings, bloating, and weight gain.",
        "Balancing progesterone and estrogen is crucial for menstrual cycle regulation.",
        "Cortisol imbalances can disrupt sleep and increase anxiety.",
        "Seed cycling is a natural way to support hormonal balance in women.",
        "Diet plays a major role in optimizing thyroid function and hormone health."
    ]
}

def fetch_pubmed_articles(keyword: str, max_results=5) -> List[str]:
    try:
        handle = Entrez.esearch(db="pubmed", term=keyword, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        pmids = record["IdList"]
        articles = []
        if pmids:
            fetch_handle = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="text")
            articles = fetch_handle.read().split("\n")
            fetch_handle.close()
        return articles
    except:
        return []


# Belgeleri güncelle
for category in categorized_documents:
    categorized_documents[category].extend(fetch_pubmed_articles(category))        


class HealthAgent:
    def __init__(self, name: str, docs: List[str]):
        self.name = name
        self.documents = docs
        self.index = self.build_faiss_index()

    def build_faiss_index(self):
        embeddings = self.get_embeddings(self.documents)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype("float32"))
        return index

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        return embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def retrieve_info(self, query: str, top_k=3) -> List[str]:
        query_embedding = self.get_embeddings([query])
        distances, indices = self.index.search(query_embedding.astype("float32"), top_k)
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]

    def generate_response(self, query: str) -> str:
        relevant_docs = self.retrieve_info(query)
        if not relevant_docs:
            return "No relevant data found."
        prompt = (
            f"You are a {self.name} expert. Use the following insights to answer the user's question.\n"
            f"Insights:\n{' '.join(relevant_docs)}\n\n"
            f"User query: {query}\n\nResponse:"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a {self.name} expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content

agents = {cat: HealthAgent(cat, docs) for cat, docs in categorized_documents.items()}

# Categorize user query
def categorize_question(query: str) -> List[str]:
    categories = list(agents.keys())
    prompt = (
        f"You are a holistic health expert. Categorize the user query into relevant topics.\n"
        f"Categories: {', '.join(categories)}\n"
        f"User Query: {query}\n"
        f"Respond with category names separated by commas."
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a holistic health expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=20
    )
    return [c.strip() for c in response.choices[0].message.content.split(",") if c in agents]

# Multi-agent response synthesis
def collaborative_response(query: str) -> str:
    categories = categorize_question(query)
    responses = {cat: agents[cat].generate_response(query) for cat in categories}
    combined = ""
    for cat, resp in responses.items():
        combined += f"### {cat}\n{resp}\n\n"
    return combined if combined else "Sorry, I couldn’t generate a helpful response."

# --- Streamlit UI ---
st.set_page_config(page_title="Holistic Health AI", layout="centered")
st.title("🌿 Holistic Health AI Assistant")
st.write("Ask anything about functional medicine, detox, sleep, or nutrition!")

query = st.text_input("Your question:")

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Get Answer") and query:
    with st.spinner("Thinking..."):
        response = collaborative_response(query)
        st.session_state.history.append((query, response))
        st.success(response)

# History
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕑 Chat History")
    for q, r in reversed(st.session_state.history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {r}")