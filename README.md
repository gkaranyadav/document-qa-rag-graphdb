# document-qa-rag-graphdb
Document Q&amp;A system using RAG and GraphDB — extracts and embeds internal documents (Mistral via Ollama), indexes with FAISS, and answers text/voice queries with page references and confidence scores. Cuts manual document search time by ~40%.


# DocuGraph AI

A little project I built to play around with combining RAG and knowledge
graphs — upload a PDF, ask it questions, and it builds up a Neo4j graph of
entities/relationships as you go, instead of just spitting out a plain
text answer like a normal RAG bot.

Runs on Streamlit. Embeddings are local (sentence-transformers), so you
don't need a GPU or a paid API for that part — the only external call is
to Groq for the actual answer generation and for pulling entities out of
the text.

## What it does

- Upload a PDF (handles scanned PDFs too, falls back to OCR if there's no
  extractable text)
- Ask questions about it, gets answers grounded in the actual document
- Every time you ask something, it also tries to extract entities and
  relationships from the relevant chunks and merges them into a graph
- You can look at the graph building up in real time on the second tab
- Voice output if you want it (just uses gTTS, nothing fancy)
- There's a button to auto-generate some starter questions if you don't
  know what to ask

## Stack

- Streamlit for the UI
- sentence-transformers + FAISS for retrieval
- Groq (Llama 3.3 70B) for generation and the entity extraction
- Neo4j for the graph itself
- pyvis for rendering it
- PyPDF2 / pytesseract / pdf2image for getting text out of PDFs

## Running it yourself

```bash
git clone https://github.com/<your-username>/<repo>.git
cd <repo>
pip install -r requirements.txt
```

You'll need a `.streamlit/secrets.toml` file with:

```toml
GROQ_API_KEY = "..."
NEO4J_URI = "neo4j+s://..."
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "..."
```

Groq keys are free at console.groq.com/keys. For Neo4j I'm using their
free AuraDB tier (neo4j.com/cloud/aura-free) so I don't have to run
anything locally.

Then just:

```bash
streamlit run app.py
```

## Deploying

Pushed this to Streamlit Community Cloud — connect the repo, point it at
`app.py`, drop the same secrets into the app's settings, done. Anyone with
the link can use it without installing anything, which was the whole
point.

## Notes / things I'd improve later

- Entity extraction quality depends a lot on how the PDF chunks split —
  sometimes get weird half-entities from paragraph breaks
- Confidence score is a rough heuristic based on FAISS distance, not a
  real calibrated probability, just useful as a relative signal
- Could probably dedupe entities better (currently matches on exact name
  string, so "Amazon" and "Amazon.com" show up as separate nodes)

## License

MIT
