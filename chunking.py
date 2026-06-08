from dataset import MetaMathQA
import re

class MathChunker:
    MAX_CHUNK_TOKEN = 256
    OVERLAP_CHARS = 80

    def __init__(self, sample, sample_id):
        self.sample = sample
        self.sample_id = sample_id
        # 'original_question_vi', 'query_vi', 'response_vi', 'type'
        org_query = (sample.get('original_question_vi'))
        query = (sample.get('query_vi'))
        response = (sample.get('response_vi'))
        math_type = (sample.get('type'))

        if not query and not response:
            return []

        chunks = []
        # Query chunk
        if query:
            chunks.append({
                'text': f'[ĐỀ BÀI] {query}',
                'chunk_type': 'question',
                'sample_id': sample_id,
                'math_type': math_type,
                'query_ref': query[:120]
            })

        # Response chunk
        if response:
            steps = self._split_solution(response)
            for step_idx, step_text in enumerate(steps):
                chunks.append({
                    'text': f'[LỜI GIẢI - bước {step_idx + 1}/{len(steps)}] {step_text}',
                    'chunk_type': 'solution',
                    'sample_id': sample_id,
                    'math_type': math_type,
                    'step_idx': step_idx,
                    'query_ref': query[:120]
                })

        if query and response:
            full_text = f'[FULL] {query} | {response}'
            if len(full_text) > 512:
                full_text = full_text[:509] + '...'
            chunks.append({
                'text': full_text,
                'chunk_type': 'full',
                'sample_id': sample_id,
                'math_type': math_type,
                'query_ref': query[:120]
            })

    def _split_solution(self, response):
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

        parts = step_pattern.split(response)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) <= 1:
            parts = self._sliding_window(response)

        merged = []
        for p in parts:
            if merged and len(p) < 30:
                merged[-1] += ' ' + p
            else:
                merged.append(p)

        return merged if merged else [response]

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

