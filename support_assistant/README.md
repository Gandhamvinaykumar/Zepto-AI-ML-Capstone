# Support Assistant

This module is a small Zepto policy assistant. It searches the local policy files with sentence-transformer embeddings, stores the vectors in ChromaDB, and uses LangGraph to choose how to answer. Mock mode is on by default, so no API key is needed.

## Setup

From the repo root:

```bash
pip install -r requirements.txt
cd support_assistant
python main.py
```

Then call the local FastAPI app:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d '{"query":"What is the delivery fee below INR 149?"}'
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d '{"query":"What is the capital of France?"}'
```

## How it works

The request goes through four steps:

- The eight policy files are read from `support_assistant/docs`.
- `all-MiniLM-L6-v2` turns the documents and question into embeddings.
- ChromaDB returns the three closest documents.
- Policy questions use the retrieved text; unrelated questions receive a short policy-only response.

`MOCK_LLM` controls the generation branch. Leave it unset, or set it to `1`, to use the local response path. Setting it to `0` selects the optional real-LLM path in the code.

## Example responses (MOCK_LLM default)

```json
{"answer":"According to the policy: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes ...","sources":["doc_01.txt"],"confidence":1.0}
```

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```
