# Meta Math QA

- Building a RAG System as a Mathematics assistant 

```text
Flow:
Raw data (Meta Math) -> Chunking -> Embedding
                                        |
                                        |               |--> Answer (If score > min_Score)
                                        v               |
        Indexing and Vector Database Storing (Qdrant) --|
                                        ^               |               
                                        | Retrieval     |--> Trigger Web Searcher (Online Search)
                                        |  
Query -> Embedding -> MathRAG ----------- 

```

```text
MetaMathQA
|---- chunking.py : Chunking the math context from dataset
|---- embedder.py : Embedding Model
|---- Fine Tune (Just practice -> No evaluation)
           |--- preprocessing.py : Clean text, format to Agent Prompt, Byte Pair Encoding + Tokenizer
           |--- train.py : Training process
|---- RAG (Meta Math Assistant)
           |--- vector_store.py : Initial Qdrant vector store
           |--- indexing.py : Indexing the embedding chunks from dataset -> vector database storing.
           |--- rag.py : MathRAG assistant (Local retrieval and WebSearcher)
           |--- app.py : Implement via FastAPI. 
``` 

---
## 1. Dataset:
- Using `5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated` dataset
```python
class MetaMathQA:
    def __init__(self, data_path='dataset'):
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            self._download_dataset()

        self.dataset = load_from_disk(self.data_path)

    def get_dataset(self):
        return self.dataset

    def _download_dataset(self):
        dataset = load_dataset('5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated', split='train')
        dataset.save_to_disk(self.data_path)
        os.makedirs(self.data_path, exist_ok=True)
        dataset.save_to_disk(self.data_path)

    def __len__(self):
        return len(self.dataset)

    def get_item_vi(self, index):
        row = self.dataset[index]
        target = ['original_question_vi', 'query_vi', 'response_vi', 'type']
        result = {k: row[k] for k in target}
        return result

    def get_item_en(self, index):
        row = self.dataset[index]
        target = ['original_question_en', 'query_en', 'response_en', 'type']
        result = {k: row[k] for k in target}
        return result
```
- This dataset contains vietnamese translated and english version of each problem.

---

## 2. Chunking and Embedding:
### a. Chunking:
- There are two different types of chunk:
  - Query chunk: problem content:
    ```python
    'text': f'[ĐỀ BÀI] {query}',
    'chunk_type': 'question',
    'sample_id': sample_id,
    'math_type': math_type,
    'query_ref': query[:120]
    ```
    - Response chunk: problem solution:
    ```python
    'text': f'[LỜI GIẢI - bước {step_idx + 1}/{len(steps)}] {step_text}',
    'chunk_type': 'solution',
    'sample_id': sample_id,
    'math_type': math_type,
    'step_idx': step_idx,
    'query_ref': query[:120]
    ```
- split the solution:
  - using step pattern to split the responses (solutions)
    ```python
    step_pattern = re.compile(
        r'(?=Bước\s*\d|'  # "Bước 1", "Bước 2"
        r'\n\s*\d+[\.\)]\s|'  # "1. " hoặc "1) "
        r'\nVậy\s|'  # "Vậy ..."
        r'\nDo đó\s|'  # "Do đó ..."
        r'\nTa có\s|'  # "Ta có ..."
        r'\nGiải:|'  # "Giải:"
        r'\nThay\s)',  # "Thay vào..."
        re.UNICODE,
    )
    ```
  - If there are no step pattern in response -> use sliding window
    ```python
    def _sliding_window(self, text):
        size = self.MAX_CHUNK_TOKEN
        overlap = self.OVERLAP_CHARS
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunk = text[start:end]

            if end < len(text):
                cut = max(
                    chunk.rfind(". "),
                    chunk.rfind(".\n"),
                    chunk.rfind("! "),
                    chunk.rfind("? "),
                )
                if cut > size // 2:
                    chunk = chunk[:cut + 1]
                    end = start + cut + 1
            chunks.append(chunk.strip())
            start = end - overlap
        return chunks
    ```
### b. Embedding:
- Using `intfloat/multilingual-e5-large` from **Sentence Transformer**:
```python
class MetaMathEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-large",
            device=device,
        )

    def embed(self, texts):
        prefixed = [f'passage: {t}' for t in texts]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=64,
        )
        return embeddings.tolist()

    def embed_query(self, query):
        embedding = self.model.encode(
            f'query: {query}',
            normalize_embeddings=True,
        )
        return embedding.tolist()
```

---
## 3. Indexing and Vector Database Store:
### a. Qdrant:
- Create Qdrant as a vector database store by Docker
- The embedded chunks will be store with `[ids, vectors, payloads]` from **PointStruct** of the chunk.
- The searching will get the point (which store above) and return cosine similarity score from the collection.
### b. Indexing:
- Index the embedded chunk before upsert into qdrant.
- Indexing flow:
```text
raw data -> chunking     Indexing the data (with index in batch chunks) 
                |                   ^                        |
                |                   |                    Embedded 
                |                   |                        |
                v                   |                        v
         save to chunks checkpoint folder    upsert chunks into vector store
```
- chunks when upsert:
```python
self.store.upsert_chunks(
    ids=list(range(start_idx, end_idx)),
    vectors=batch_vectors,
    payloads=current_batch_chunks,
)
```
- where:
  - **batch_vectors**: is an list of text in chunks of batch which embedded
  ```python
    batch_texts = [c['text'] for c in current_batch_chunks]
    batch_vectors = self.embedder.embed(batch_texts)
  ```
  - **ids**: list contains range of start idx and end idx:
  ```python
    start_idx = done_count + i
    end_idx = start_idx + actual_batch_len
  ```
  - **payloads**: Current batch chunks (The current chunk ids -> batch-size steps)

---

## 4. RAG:
### a. Web Searcher:
### b. Math Searcher (Qdrant retrieval):

---

## 5. Demo:


---
## Fine Tuning Practicing:
### 1. Preprocessing:
### 2. Training

#### Currently, I have not used any evaluation metrics for this fine-tuning so just take it easy.

