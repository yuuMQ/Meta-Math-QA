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


