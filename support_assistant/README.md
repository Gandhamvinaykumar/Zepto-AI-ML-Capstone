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

The first request downloads `all-MiniLM-L6-v2` if it is not already cached. After that, the vectors are reused from the local `chroma_store` directory.

## How it works

The request follows this path:

```text
docs/*.txt
	-> PolicyAssistant._bootstrap_collection()
	-> SentenceTransformer("all-MiniLM-L6-v2")
	-> ChromaDB collection: zepto_policies
	-> classify_intent node
	   -> policy_question: retrieve_and_answer node -> JSON response
	   -> general_question: direct_answer node -> JSON response
```

1. **Ingestion:** `PolicyAssistant._bootstrap_collection()` reads the eight files under `support_assistant/docs`. Each document is used as one chunk and stored with its filename as the ChromaDB ID.
2. **Embedding:** `SentenceTransformer("all-MiniLM-L6-v2")` embeds each document and each incoming query locally. ChromaDB stores the vectors in the persistent `zepto_policies` collection under `chroma_store`.
3. **Retrieval:** the LangGraph `retrieve_and_answer` node embeds policy questions and asks ChromaDB for the top three matching documents.
4. **Generation:** in the default mock mode, `retrieve_and_answer` returns the required `Based on the retrieved context: ...` response and `direct_answer` returns the fixed policy-only message. With `MOCK_LLM=0`, the retrieved context is inserted into `PROMPT_TEMPLATE` and passed to the optional real-LLM hook; its response is validated by `AnswerResponse`.

The graph is built in `build_graph()` with three nodes: `classify_intent`, `retrieve_and_answer`, and `direct_answer`. A conditional edge sends `policy_question` to retrieval and `general_question` directly to the canned response. The keyword classifier and both mock answer branches make no LLM call when `MOCK_LLM` is unset or set to `1`.

The response always follows the Pydantic schema `answer` (string), `sources` (list of document IDs), and `confidence` (0 to 1). The optional real-LLM answer path tries the initial prompt plus two corrective prompts when the returned JSON fails validation; after three failures it returns a marked error response with confidence `0.0`.

`MOCK_LLM` controls the generation branch. Leave it unset, or set it to `1`, to use the local response path. Setting it to `0` selects the optional real-LLM path in the code.

## Example responses (MOCK_LLM default)

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes ...","sources":["doc_01.txt","doc_03.txt","doc_05.txt"],"confidence":1.0}
```

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## Docker

Build from the repository root so the image can copy the shared dependency file:

```bash
docker build -f support_assistant/Dockerfile -t zepto-support-assistant .
docker run --rm -p 7860:7860 zepto-support-assistant
```

Then open `http://127.0.0.1:7860/docs` or send a `POST /ask` request as shown above. The root `Dockerfile` provides the same service with the shorter `docker build -t zepto-capstone .` command.

The module Docker image starts Uvicorn directly on port 7860 and serves the same `/ask` endpoint as the local Python command.
