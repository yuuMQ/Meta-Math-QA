from datasets import concatenate_datasets, load_from_disk
from chunking import MathChunker
from embedder import MetaMathEmbedder
from vector_store import QDrantVectorStore
from dataset import MetaMathQA
import torch
from tqdm import tqdm
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHUNK_CHECKPOINT = 'checkpoints/chunks.json'
EMBED_CHECKPOINT = 'checkpoints/embedded_count.txt'

class IndexBuilder:
    def __init__(self):
        self.embedder = MetaMathEmbedder()
        self.store = QDrantVectorStore()
        self.chunker = MathChunker()

    def build_from_dataset(self):
        dataset = MetaMathQA().get_dataset()

        print(f'[INDEX] Tổng: {len(dataset)} samples')

        if os.path.exists(CHUNK_CHECKPOINT):
            print('Load Chunk Checkpoint')
            with open(CHUNK_CHECKPOINT, 'r', encoding='utf-8') as f:
                all_chunks = json.load(f)
            print(f'[INDEX] load {len(all_chunks)} chunks')

        else:
            os.makedirs('checkpoints', exist_ok=True)
            print('[Index] Chunking theo cấu trúc đề bài / lời giải ...')
            all_chunks = []
            for i, sample in tqdm(enumerate(dataset), desc='Chunking !!!', total=len(dataset)):
                chunks = self.chunker.chunk_sample(sample, sample_id=i)

                all_chunks.extend(chunks)

            print('Save to chunk checkpoint !!!')
            with open(CHUNK_CHECKPOINT, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, ensure_ascii=False)
            print(f'Save Chunk Checkpoint Done: {len(all_chunks)}!!!')

        done_count = 0
        if os.path.exists(EMBED_CHECKPOINT):
            with open(EMBED_CHECKPOINT, 'r', encoding='utf-8') as f:
                done_count = int(f.read().strip())
            print(f'Resume embedding từ chunk {done_count:,} !!!')

        # question_count = sum(1 for c in all_chunks if c['chunk_type'] == 'question')
        # solution_count = sum(1 for c in all_chunks if c['chunk_type'] == 'solution')
        # full_count = sum(1 for c in all_chunks if c['chunk_type'] == 'full')
        #
        # print(f'question: {question_count:,} | solution: {solution_count:,} | full: {full_count:,}`')

        # remaining_texts = [c['text'] for c in all_chunks[done_count:]]
        remaining_chunks = all_chunks[done_count:]

        if remaining_chunks:
            print(f'[INDEX] Xử lý {len(remaining_chunks):,} chunks còn lại trên {device}...')

            batch_size = 256
            for i in tqdm(range(0, len(remaining_chunks), batch_size), desc='Embedding'):
                current_batch_chunks = remaining_chunks[i: i + batch_size]
                actual_batch_len = len(current_batch_chunks)

                start_idx = done_count + i
                end_idx = start_idx + actual_batch_len

                batch_texts = [c['text'] for c in current_batch_chunks]
                batch_vectors = self.embedder.embed(batch_texts)

                self.store.upsert_chunks(
                    ids=list(range(start_idx, end_idx)),
                    vectors=batch_vectors,
                    payloads=current_batch_chunks,
                )

                with open(EMBED_CHECKPOINT, 'w', encoding='utf-8') as f:
                    f.write(str(end_idx))

        print(f'[INDEX] đã index {len(all_chunks)} chunks vào Qdrant vector store')

if __name__ == '__main__':
    builder = IndexBuilder()
    builder.build_from_dataset()

