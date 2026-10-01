"""MVP: Agent IA finance internationale (collecte + analyse + sentiment + RAG + LangGraph + Streamlit)
Lancer:  streamlit run app.py
"""
import os, sqlite3
from typing import TypedDict

import chromadb
import feedparser
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

DB = "finance.db"
WB = "https://api.worldbank.org/v2"
INDICATORS = {
    "Croissance PIB (%)": "NY.GDP.MKTP.KD.ZG",
    "Inflation (%)": "FP.CPI.TOTL.ZG",
    "Chômage (%)": "SL.UEM.TOTL.ZS",
}
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
# Pour Ollama (gratuit, local): OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama LLM_MODEL=mistral

PROMPT = """Tu es un assistant financier. Réponds UNIQUEMENT à partir du contexte ci-dessous.
Si l'information n'y est pas, dis-le clairement. Cite la source et la date. Aucun conseil d'investissement.

Contexte:
{contexte}

Question: {question}
Réponse:"""


# ---------- Collecte & stockage ----------
def db():
    return sqlite3.connect(DB)


def collect_macro(iso3: str):
    frames = []
    for nom, code in INDICATORS.items():
        r = requests.get(f"{WB}/country/{iso3}/indicator/{code}",
                         params={"format": "json", "date": "2015:2025", "per_page": 100}, timeout=15)
        r.raise_for_status()
        for d in (r.json()[1] or []):
            if d["value"] is not None:
                frames.append({"pays": iso3, "indicateur": nom, "annee": int(d["date"]),
                               "valeur": d["value"], "source": "World Bank"})
    df = pd.DataFrame(frames)
    with db() as c:
        df.to_sql("macro", c, if_exists="replace", index=False)
    return df


def collect_prices(ticker: str):
    h = yf.Ticker(ticker).history(period="6mo")[["Close", "Volume"]].reset_index()
    h["Date"] = h["Date"].astype(str).str[:10]
    h["ticker"] = ticker
    h["rendement"] = h["Close"].pct_change()
    z = (h["rendement"] - h["rendement"].mean()) / h["rendement"].std()
    h["anomalie"] = z.abs() > 2  # Z-score
    with db() as c:
        h.to_sql("prix", c, if_exists="replace", index=False)
    return h


@st.cache_resource
def chroma_col():
    return chromadb.Client().get_or_create_collection("news")


def collect_news(ticker: str):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    items = feedparser.parse(url).entries[:15]
    col = chroma_col()
    for i, e in enumerate(items):
        col.upsert(ids=[f"{ticker}-{i}"], documents=[f"{e.title}. {e.get('summary', '')}"],
                   metadatas=[{"source": e.get("link", "Yahoo Finance"), "date": e.get("published", "")}])
    return len(items)


# ---------- Sentiment (FinBERT) ----------
@st.cache_resource
def finbert():
    from transformers import pipeline
    return pipeline("text-classification", model="ProsusAI/finbert")


# ---------- LangGraph ----------
class Etat(TypedDict):
    question: str
    ticker: str
    pays: str
    contexte: str
    reponse: str


def router(e: Etat) -> str:
    q = e["question"].lower()
    if any(k in q for k in ["sentiment", "actualité", "news", "اخبار"]):
        return "sentiment"
    if any(k in q for k in ["volatil", "anomal", "cours", "prix", "clôture"]):
        return "analyse"
    return "conversationnel"


def n_analyse(e: Etat):
    with db() as c:
        p = pd.read_sql("select * from prix", c)
    vol = p["rendement"].std() * (252 ** 0.5)
    anom = p[p["anomalie"]]["Date"].tolist()
    ctx = (f"[yfinance] {e['ticker']} dernier cours {p['Close'].iloc[-1]:.2f} au {p['Date'].iloc[-1]}. "
           f"Volatilité annualisée: {vol:.1%}. Jours anormaux (|z|>2): {anom}")
    return {"contexte": ctx}


def n_sentiment(e: Etat):
    docs = chroma_col().query(query_texts=[e["question"]], n_results=5)["documents"][0]
    try:  # FinBERT si installé (PC), sinon le LLM évalue le sentiment (mobile/cloud)
        scores = finbert()([d[:500] for d in docs])
        ctx = "\n".join(f"[Yahoo RSS] {d[:200]} -> {s['label']} ({s['score']:.2f})" for d, s in zip(docs, scores))
    except ImportError:
        ctx = "Titres (évalue leur sentiment positif/négatif/neutre):\n" + "\n".join(f"[Yahoo RSS] {d[:200]}" for d in docs)
    return {"contexte": ctx}


def n_conv(e: Etat):
    res = chroma_col().query(query_texts=[e["question"]], n_results=4)
    news = "\n".join(f"[{m['source']}] {d[:300]}" for d, m in zip(res["documents"][0], res["metadatas"][0]))
    with db() as c:
        m = pd.read_sql("select indicateur, annee, valeur from macro order by annee desc", c).head(9)
    return {"contexte": f"Macro (World Bank, {e['pays']}):\n{m.to_string(index=False)}\n\nActualités:\n{news}"}


def n_generate(e: Etat):
    r = OpenAI().chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role": "user", "content": PROMPT.format(contexte=e["contexte"], question=e["question"])}])
    return {"reponse": r.choices[0].message.content}


@st.cache_resource
def build_graph():
    g = StateGraph(Etat)
    g.add_node("analyse", n_analyse)
    g.add_node("sentiment", n_sentiment)
    g.add_node("conversationnel", n_conv)
    g.add_node("generate", n_generate)
    g.add_conditional_edges(START, router, {k: k for k in ["analyse", "sentiment", "conversationnel"]})
    for n in ["analyse", "sentiment", "conversationnel"]:
        g.add_edge(n, "generate")
    g.add_edge("generate", END)
    return g.compile()


# ---------- Interface ----------
st.set_page_config(page_title="Agent IA Finance", layout="wide")
st.title("Agent IA - Suivi de la finance internationale")
ticker = st.sidebar.text_input("Ticker", "AAPL")
pays = st.sidebar.text_input("Pays (ISO3)", "TUN")

if st.sidebar.button("Collecter les données"):
    with st.spinner("Collecte..."):
        collect_macro(pays)
        collect_prices(ticker)
        n = collect_news(ticker)
    st.sidebar.success(f"OK ({n} articles)")

tab1, tab2 = st.tabs(["Tableau de bord", "Assistant"])
with tab1:
    try:
        with db() as c:
            p = pd.read_sql("select * from prix", c)
            m = pd.read_sql("select * from macro", c)
        st.line_chart(p.set_index("Date")["Close"])
        st.write("Anomalies (Z-score):", p[p["anomalie"]][["Date", "Close", "rendement"]])
        st.dataframe(m.pivot(index="annee", columns="indicateur", values="valeur"))
    except Exception:
        st.info("Clique sur « Collecter les données » d'abord.")

with tab2:
    q = st.chat_input("Pose ta question (ex: Quelle est la volatilité ?)")
    if q:
        st.chat_message("user").write(q)
        out = build_graph().invoke({"question": q, "ticker": ticker, "pays": pays, "contexte": "", "reponse": ""})
        st.chat_message("assistant").write(out["reponse"])
        with st.expander("Contexte utilisé (sources)"):
            st.text(out["contexte"])
