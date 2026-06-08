from datasets import concatenate_datasets, load_from_disk
from chunking import MathChunker
from embedder import MetaMathEmbedder
from vector_store import QDrantVectorStore
from dataset import MetaMathQA

class IndexBuilder:
    def __init__(self):
        self.embedder = MetaMathEmbedder()
        self.store = QDrantVectorStore()
        self.chunker = MathChunker()

    def build_from_dataset(self):
        dataset = MetaMathQA().get_dataset()

        print(f'[INDEX] Tổng: {len(dataset)} samples')

        print('[Index] Chunking theo cấu trúc đề bài / lời giải ...')
        all_chunks = []

        for i, sample in enumerate(dataset):
            chunks = self.chunker.chunk_sample(sample, sample_id=i)

            all_chunks.extend(chunks)
            if i % 5000 == 0 and i > 0:
                print(f'sample={i:,} | chunks={chunks} | lens={len(chunks):,}')

        print(f'[INDEX] {len(all_chunks)} chunks')
        question_count = sum(1 for c in all_chunks if c['chunk_type'] == 'question')
        solution_count = sum(1 for c in all_chunks if c['chunk_type'] == 'solution')
        full_count = sum(1 for c in all_chunks if c['chunk_type'] == 'full')

        print(f'question: {question_count:,} | solution: {solution_count:,} | full: {full_count:,}`')

        texts = [c['text'] for c in all_chunks]
        batch_size = 64

        all_vectors, all_ids = [], []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vectors = self.embedder.embed(texts=batch)
            all_vectors.extend(vectors)
            all_ids.extend(range(i, i + len(batch)))
            if (i // batch_size) % 20 == 0:
                print(f'{i + len(batch):,} / {len(texts):,}')

        self.store.upsert_chunks(all_ids, all_vectors, all_chunks)
        print(f'[INDEX] đã index {len(all_chunks)} chunks vào Qdrant vector store')

if __name__ == '__main__':
    builder = IndexBuilder()
    builder.build_from_dataset()

