import streamlit as st
import tempfile, os, base64, re, json
from PyPDF2 import PdfReader
import pytesseract
import pdf2image
from neo4j import GraphDatabase
from groq import Groq
from gtts import gTTS
import plotly.graph_objects as go
from pyvis.network import Network
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# rename this if you want, just changes the page title / bot name
APP_TITLE = "DocuGraph AI"
ASSISTANT_NAME = "Aura"

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
MIN_PARAGRAPH_LEN = 20
TOP_K = 3

NEO4J_URI = st.secrets.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = st.secrets.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = st.secrets.get("NEO4J_PASSWORD", "password")


def distance_to_confidence(d):
    # faiss gives L2 distance, not confidence, so map it to something
    # that actually reads like a percentage. thresholds tuned by trial and error
    if d < 0.3:
        c = 0.92 + (0.3 - d) * 0.27
    elif d < 0.6:
        c = 0.85 + (0.6 - d) * 0.23
    elif d < 1.0:
        c = 0.75 + (1.0 - d) * 0.25
    elif d < 1.5:
        c = 0.65 + (1.5 - d) * 0.2
    else:
        c = 0.55 + (2.0 - d) * 0.2
    return max(0.55, min(0.99, c))


class VectorStore:
    """holds the chunk embeddings + faiss index for one uploaded doc"""

    def __init__(self):
        try:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            st.error(f"couldn't load embedding model: {e}")
            self.embedder = None
        self.index = None
        self.chunks = []
        self.chunk_meta = []

    def add_documents(self, chunks, meta):
        self.chunks = chunks
        self.chunk_meta = meta
        if not chunks or not self.embedder:
            return
        vecs = self.embedder.encode(chunks)
        self.index = faiss.IndexFlatL2(vecs.shape[1])
        self.index.add(np.array(vecs))

    def search(self, query, k=TOP_K):
        if not self.chunks or self.index is None:
            return []
        qvec = self.embedder.encode([query])
        dists, idxs = self.index.search(np.array(qvec), k)
        out = []
        for i, idx in enumerate(idxs[0]):
            if idx >= len(self.chunks):
                continue
            out.append({
                "content": self.chunks[idx],
                "metadata": self.chunk_meta[idx],
                "similarity": round(distance_to_confidence(dists[0][i]), 3),
            })
        out.sort(key=lambda x: x["similarity"], reverse=True)
        return out

    def sample_text(self, n=5):
        if not self.chunks:
            return ""
        return " ".join(self.chunks[:n])


class GraphExtractor:
    """asks the LLM to pull entities/relationships out of the retrieved chunks"""

    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        except Exception as e:
            st.error(f"groq init failed: {e}")
            self.client = None

    def extract(self, chunks):
        if not self.client or not chunks:
            return [], []

        text = "\n\n".join(c["content"] for c in chunks)
        prompt = f"""Extract a knowledge graph from this text:

{text}

Return JSON only, nothing else:
{{"entities": [{{"name": "...", "type": "PERSON|ORGANIZATION|CONCEPT|PRODUCT|EVENT|TECHNOLOGY"}}],
 "relationships": [{{"source": "...", "target": "...", "type": "IMPACTS|CAUSES|INVESTS_IN|DEVELOPS|COMPETES_WITH|PARTNERS_WITH"}}]}}"""

        try:
            resp = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You extract knowledge graphs from text. Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1500,
            )
            raw = resp.choices[0].message.content
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return [], []
            data = json.loads(m.group())
            return data.get("entities", []), data.get("relationships", [])
        except Exception as e:
            st.warning(f"graph extraction failed: {e}")
            return [], []


class GraphProcessor:
    def __init__(self):
        self.vector_store = VectorStore()
        self.extractor = GraphExtractor()
        self.current_doc = None
        self.driver = None
        self._connect()

    def _connect(self):
        try:
            uri = NEO4J_URI.replace("https://", "bolt://").replace("http://", "bolt://")
            self.driver = GraphDatabase.driver(uri, auth=(NEO4J_USER, NEO4J_PASS))
            with self.driver.session() as s:
                s.run("RETURN 1")
        except Exception as e:
            st.error(f"couldn't connect to neo4j: {e}")
            self.driver = None

    def _is_scanned(self, path):
        reader = PdfReader(path)
        if len(reader.pages) == 0:
            return False
        text_pages = sum(1 for p in reader.pages if len((p.extract_text() or "").strip()) > 50)
        return (text_pages / len(reader.pages)) <= 0.5

    def process_pdf(self, uploaded_file):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(uploaded_file.getvalue())
        tmp.close()

        try:
            if self._is_scanned(tmp.name):
                pages = pdf2image.convert_from_path(tmp.name, dpi=200)
                extracted = [(i + 1, pytesseract.image_to_string(p)) for i, p in enumerate(pages)]
            else:
                reader = PdfReader(tmp.name)
                extracted = [(i, p.extract_text() or "") for i, p in enumerate(reader.pages, 1)]

            self.current_doc = uploaded_file.name
            with self.driver.session() as s:
                s.run("MATCH (d:Document {name: $n}) DETACH DELETE d", n=self.current_doc)
                s.run("CREATE (d:Document {name: $n})", n=self.current_doc)

            chunks, meta = [], []
            for page_num, text in extracted:
                for para in text.split("\n\n"):
                    para = para.strip()
                    if len(para) >= MIN_PARAGRAPH_LEN:
                        chunks.append(para)
                        meta.append({"page": page_num, "document": self.current_doc})

            self.vector_store.add_documents(chunks, meta)
            return len(chunks)
        finally:
            os.unlink(tmp.name)

    def answer_question(self, question):
        if not self.driver or not self.current_doc:
            return [], "no document loaded"

        chunks = self.vector_store.search(question)
        if not chunks:
            return [], "nothing relevant found"

        entities, rels = self.extractor.extract(chunks)
        if entities or rels:
            self._save_to_graph(entities, rels, chunks)

        return chunks, f"+{len(entities)} entities, +{len(rels)} relationships"

    def _save_to_graph(self, entities, rels, source_chunks):
        pages = sorted(set(c["metadata"]["page"] for c in source_chunks))
        with self.driver.session() as s:
            for e in entities:
                s.run(
                    "MERGE (n:Entity {name: $name, document: $doc}) SET n.type = $type",
                    name=e.get("name", ""), doc=self.current_doc, type=e.get("type", "CONCEPT"),
                )
            for r in rels:
                # skip self-loops, they show up sometimes from bad extractions
                s.run(
                    """MATCH (a:Entity {name: $src, document: $doc})
                       MATCH (b:Entity {name: $tgt, document: $doc})
                       WHERE a <> b
                       MERGE (a)-[rel:RELATED_TO {type: $rtype}]->(b)
                       SET rel.source_pages = $pages""",
                    src=r.get("source", ""), tgt=r.get("target", ""),
                    doc=self.current_doc, rtype=r.get("type", "related_to"), pages=pages,
                )

    def stats(self):
        if not self.driver or not self.current_doc:
            return 0, 0
        with self.driver.session() as s:
            row = s.run(
                """MATCH (d:Document {name: $doc})
                   OPTIONAL MATCH (e:Entity {document: $doc})
                   OPTIONAL MATCH (e)-[r:RELATED_TO]->(:Entity)
                   RETURN count(DISTINCT e) AS ents, count(DISTINCT r) AS rels""",
                doc=self.current_doc,
            ).single()
            return row["ents"] or 0, row["rels"] or 0

    def graph_data(self, limit=100):
        if not self.driver or not self.current_doc:
            return None
        with self.driver.session() as s:
            result = s.run(
                """MATCH (e:Entity {document: $doc})
                   OPTIONAL MATCH (e)-[r:RELATED_TO]-(o:Entity)
                   RETURN e, r, o LIMIT $limit""",
                doc=self.current_doc, limit=limit,
            )
            seen, nodes, edges = set(), [], []
            for row in result:
                for n in (row["e"], row["o"]):
                    if n and n.id not in seen:
                        nodes.append({"id": n.id, "label": n.get("name", "?"), "type": n.get("type", "CONCEPT")})
                        seen.add(n.id)
                if row["r"] and row["e"] and row["o"]:
                    edges.append({"source": row["e"].id, "target": row["o"].id, "type": row["r"].get("type", "")})
            return {"nodes": nodes, "edges": edges}


def groq_client():
    try:
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except Exception as e:
        st.error(f"groq init failed: {e}")
        return None


def generate_answer(client, question, chunks):
    if not chunks:
        return f"couldn't find anything relevant for '{question}'", 0.0

    sims = [c["similarity"] for c in chunks]
    weights = [0.5, 0.3, 0.2][:len(sims)]
    if len(weights) != len(sims):
        weights = [1 / len(sims)] * len(sims)
    conf = max(0.6, min(0.98, sum(s * w for s, w in zip(sims, weights))))

    if not client:
        text = ". ".join(s.strip() for c in chunks for s in c["content"].split(".") if s.strip())
        return text[:600], conf

    try:
        ctx = "\n\n".join(c["content"] for c in chunks)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Answer only from the given context. If it's not there, say you can't find it in the document."},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return resp.choices[0].message.content, conf
    except Exception:
        text = ". ".join(s.strip() for c in chunks for s in c["content"].split(".") if s.strip())
        return text[:600], conf


FALLBACK_QUESTIONS = [
    "What is the main purpose of this document?",
    "What are the key findings or conclusions?",
    "Who is the intended audience for this content?",
    "What methodology or approach was used?",
    "What are the main recommendations?",
]


def suggest_questions(client, sample_text):
    if not client or not sample_text:
        return FALLBACK_QUESTIONS
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Generate 5-6 specific questions about this document content, one per line, no numbering."},
                {"role": "user", "content": sample_text},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        lines = resp.choices[0].message.content.split("\n")
        qs = [re.sub(r"^[\d\-•.\s]+", "", l).strip() for l in lines]
        qs = [q for q in qs if len(q) > 10 and "?" in q]
        return qs[:6] if qs else FALLBACK_QUESTIONS
    except Exception:
        return FALLBACK_QUESTIONS


def speak(text):
    try:
        clean = re.sub(r"[^\w\s.,?-]", "", text)[:500]
        for a, b in [("%", " percent"), ("$", " dollars"), ("&", " and")]:
            clean = clean.replace(a, b)
        tts = gTTS(text=clean, lang="en")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        b64 = base64.b64encode(open(tmp.name, "rb").read()).decode()
        st.components.v1.html(f'<audio autoplay><source src="data:audio/mp3;base64,{b64}"></audio>', height=0)
        os.unlink(tmp.name)
    except Exception as e:
        st.warning(f"tts failed: {e}")


def draw_graph(data):
    if not data or not data["nodes"]:
        st.info("ask a question first — the graph builds as you go")
        return
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.barnes_hut()
    colors = {"PERSON": "#FF6B6B", "ORGANIZATION": "#4ECDC4", "CONCEPT": "#45B7D1",
              "PRODUCT": "#96CEB4", "TECHNOLOGY": "#FFE66D", "EVENT": "#FF9F1C"}
    for n in data["nodes"]:
        net.add_node(n["id"], label=n["label"], color=colors.get(n["type"], "#999"), title=n["label"])
    for e in data["edges"]:
        net.add_edge(e["source"], e["target"], title=e["type"], color="#ccc")
    net.save_graph("kg.html")
    st.components.v1.html(open("kg.html", encoding="utf-8").read(), height=600, scrolling=True)


def show_msg(role, content, conf=0, pages=None, graph_note=""):
    cls = "user-msg" if role == "user" else "bot-msg"
    who = "You" if role == "user" else ASSISTANT_NAME
    extra = ""
    if conf:
        extra += f'<span class="badge">confidence {conf*100:.0f}%</span>'
    if pages:
        extra += f'<span class="badge">pages {", ".join(map(str, pages))}</span>'
    if graph_note:
        extra += f'<span class="badge">{graph_note}</span>'
    st.markdown(f'<div class="{cls}"><b>{who}:</b> {content} {extra}</div>', unsafe_allow_html=True)


def ask(question, proc, client, voice_on):
    st.session_state.messages.append({"role": "user", "content": question})
    show_msg("user", question)
    with st.spinner("thinking..."):
        chunks, note = proc.answer_question(question)
        answer, conf = generate_answer(client, question, chunks)
        pages = sorted(set(c["metadata"]["page"] for c in chunks)) if chunks else []
        st.session_state.messages.append({"role": "bot", "content": answer, "conf": conf, "pages": pages, "note": note})
        show_msg("bot", answer, conf, pages, note)
        if voice_on:
            speak(answer)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown("""
    <style>
    .main-title { font-size: 2.3rem; font-weight: 800; background: linear-gradient(135deg,#175CFF,#00A3FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .user-msg { background:#e6f3ff; border-left:4px solid #175CFF; padding:0.8rem; border-radius:8px; margin-bottom:0.6rem; }
    .bot-msg { background:#f0f8ff; border-left:4px solid #00A3FF; padding:0.8rem; border-radius:8px; margin-bottom:0.6rem; }
    .badge { background:#eee; border-radius:10px; padding:0.15rem 0.5rem; font-size:0.75rem; margin-left:0.4rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="main-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.caption("upload a pdf, ask it questions, watch the knowledge graph fill in")
    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.pdf_done = False
        st.session_state.suggestions = []
        st.session_state.show_suggestions = False
        st.session_state.suggest_clicked = False

    if "proc" not in st.session_state:
        st.session_state.proc = GraphProcessor()
    if "groq" not in st.session_state:
        st.session_state.groq = groq_client()

    st.sidebar.title("controls")
    up = st.sidebar.file_uploader("upload pdf", type="pdf")
    if up and not st.session_state.pdf_done:
        with st.spinner("processing..."):
            n = st.session_state.proc.process_pdf(up)
            if n:
                st.session_state.pdf_done = True

    if st.session_state.pdf_done:
        st.sidebar.success("document loaded")
    voice_on = st.sidebar.checkbox("voice answers", True)
    st.sidebar.write("neo4j:", "connected" if st.session_state.proc.driver else "not connected")

    tab1, tab2, tab3 = st.tabs(["Chat", "Knowledge Graph", "Stats"])

    with tab1:
        if not st.session_state.pdf_done:
            st.info("upload a pdf to get started")
        else:
            for m in st.session_state.messages:
                show_msg(m["role"], m["content"], m.get("conf", 0), m.get("pages"), m.get("note", ""))

            if not st.session_state.messages and not st.session_state.suggest_clicked:
                c1, c2, c3 = st.columns([1, 2, 1])
                if c2.button("suggest some questions", use_container_width=True):
                    st.session_state.suggest_clicked = True
                    sample = st.session_state.proc.vector_store.sample_text()
                    st.session_state.suggestions = suggest_questions(st.session_state.groq, sample)
                    st.session_state.show_suggestions = True
                    st.rerun()

            if st.session_state.show_suggestions and st.session_state.suggestions:
                st.write("**pick one, or just type your own below:**")
                for i, q in enumerate(st.session_state.suggestions):
                    c1, c2, c3 = st.columns([1, 3, 1])
                    if c2.button(q, key=f"sq{i}", use_container_width=True):
                        st.session_state.show_suggestions = False
                        ask(q, st.session_state.proc, st.session_state.groq, voice_on)
                        st.rerun()

            typed = st.chat_input("ask something about the document...")
            if typed:
                ask(typed, st.session_state.proc, st.session_state.groq, voice_on)

    with tab2:
        if not st.session_state.pdf_done:
            st.info("upload a pdf first")
        else:
            data = st.session_state.proc.graph_data()
            draw_graph(data)
            if data:
                st.caption(f"{len(data['nodes'])} nodes, {len(data['edges'])} edges so far")

    with tab3:
        if not st.session_state.pdf_done:
            st.info("upload a pdf first")
        else:
            ents, rels = st.session_state.proc.stats()
            c1, c2 = st.columns(2)
            c1.metric("entities", ents)
            c2.metric("relationships", rels)

            if st.session_state.proc.driver:
                with st.session_state.proc.driver.session() as s:
                    rows = s.run(
                        "MATCH (e:Entity {document: $d}) RETURN e.type AS t, count(*) AS n ORDER BY n DESC LIMIT 10",
                        d=st.session_state.proc.current_doc,
                    )
                    rows = list(rows)
                if rows:
                    fig = go.Figure([go.Bar(x=[r["t"] or "unknown" for r in rows], y=[r["n"] for r in rows], marker_color="#175CFF")])
                    fig.update_layout(height=380, title="entity types")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("no entities yet, ask a few questions first")


if __name__ == "__main__":
    main()
