import torch
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from embedder import MetaMathEmbedder
from vector_store import QDrantVectorStore

MIN_SCORE = 0.45
load_dotenv()

API_KEY = os.getenv("API_KEY")

SYSTEM_PROMPT = (
    "Bạn là trợ lý toán học thông minh. "
    "Hãy giải các bài toán chính xác, trình bày từng bước rõ ràng bằng tiếng Việt. "
    "Chỉ dùng thông tin từ ngữ cảnh được cung cấp và kiến thức toán học của bạn."
)
WEB_SYSTEM_PROMPT = (
    "Bạn là chuyên gia toán học. "
    "Với bài toán được đưa ra và kết quả tìm kiếm từ internet, hãy: "
    "1) Xác định dạng toán và công thức liên quan. "
    "2) Tóm tắt các thông tin hữu ích từ kết quả tìm kiếm. "
    "3) Gợi ý hướng giải ngắn gọn. "
    "Trả lời bằng tiếng Việt, KHÔNG giải hoàn chỉnh — chỉ cung cấp context."
)


# Web Search -> Find information of Mathematics online
class WebSearcher:
    def __init__(self):
        self.client = ChatGroq(
            model='llama-3.3-70b-versatile',
            temperature=0,
            api_key=API_KEY
        )
        self.search_tool = DuckDuckGoSearchRun()
        self.parser = StrOutputParser()

    def _raw_search(self, query):
        math_query = f'Toán học: Cách giải {query}'
        return self.search_tool.invoke(math_query)

    def _summerize(self, query, raw):
        messages = [
            SystemMessage(content=WEB_SYSTEM_PROMPT),
            HumanMessage(content=(
                f'Bài toán: {query}\n\n'
                f'Kết quả tìm kiếm từ Internet:\n{raw}\n\n'
                f'Hãy tóm tắt thông tin hữu ích để giải bài toán trên.'
            )),
        ]
        response = self.client.invoke(messages)
        return self.parser.invoke(response)

    def search_web_content(self, query):
        print(f"[WebSearcher] Searching: {query[:60]}...")

        raw = self._raw_search(query)

        if not raw.strip():
            print('[WEBSEARCHER] không có kết quả!!!')
            return self._groq_only_context(query)

        summary = self._summerize(query, raw)
        return summary

    def _groq_only_context(self, query):
        messages = [
            SystemMessage(content=WEB_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        response = self.client.invoke(messages)
        return self.parser.invoke(response)


# Local LLM -> Sử dụng Viet-Sailor-4B
class MathAssistant:
    def __init__(self, model_path='base_model/Viet-Sailor-4B-Instruct.Q8_0.gguf'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = LlamaCpp(
            model_path=model_path,
            n_ctx=32768,
            n_gpu_layers=0,
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.3,
            max_tokens=1024,
            verbose=False,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        self.parser = StrOutputParser()

    def _build_prompt(self, query, context, web_hint):
        ctx_block = f"### Ngữ cảnh từ dữ liệu:\n{context}"  if context  else ""
        web_block = f"### Gợi ý từ internet:\n{web_hint}"    if web_hint else ""
        separator = "\n\n" if ctx_block and web_block else ""

        return (
            f"[SYS]{SYSTEM_PROMPT}[/SYS]"
            f"[CTX]{ctx_block}{separator}{web_block}[/CTX]"
            f"[USR]{query}[/USR]"
            f"[AST]"
        )

    def generate(self, query, context, web_hint, max_tokens=1024):
        prompt = self._build_prompt(query, context, web_hint)

        raw = self.llm.invoke(prompt, stop=["[EOS]", "</s>", "[USR]", "[SYS]", "[/AST]", "###"])
        answer = self.parser.invoke(raw).strip()

        if '[EOS]' in answer:
            answer = answer.split('[EOS]')[0].strip()

        return answer


class MathRAG:
    def __init__(self):
        self.embedder = MetaMathEmbedder()
        self.vector_store = QDrantVectorStore()
        self.llm = MathAssistant()
        self.web_searcher = WebSearcher()

    def _build_context(self, hits):
        seen = {}
        for h in hits:
            sample_id = h.get('sample_id', -1)
            if sample_id not in seen or h['score'] > seen[sample_id]['score']:
                seen[sample_id] = h

        ordered = sorted(seen.values(), key=lambda x: (
            {'question': 0, 'solution': 1, 'full': 2}.get(x.get('chunk_type', 'full'), 3),
            -x['score'],
        ))

        labels = {
            'question': 'Bài toán tương tự',
            'solution': 'Lời giải tham khảo',
            'full': 'Bài toán hoàn chỉnh'
        }
        parts = []
        for h in ordered:
            label = labels.get(h.get("chunk_type", ""), "Tham khảo")
            parts.append(f"{label} (score={h['score']:.2f}):\n{h['text']}")

        full_context = "\n\n---\n\n".join(parts)
        if len(full_context) > 1500:
            full_context = full_context[:1500] + "..."
        return full_context

    def _build_prompt(self, query, context, web_hint):
        ctx_block = f"### Ngữ cảnh từ dữ liệu:\n{context}" if context else ""
        web_block = f"### Gợi ý phương pháp:\n{web_hint}" if web_hint else ""

        return (
            f"[SYS]{SYSTEM_PROMPT}[/SYS]"
            f"[CTX]{ctx_block}\n{web_block}[/CTX]"
            f"[USR]{query}[/USR]"
            f"[AST]"
        )

    def answer(self, query, verbose):
        query_vector = self.embedder.embed_query(query)
        hits = self.vector_store.search(query_vector, top_k=5)

        if verbose:
            print(f"[RAG] Retrieved {len(hits)} chunks (min_score={MIN_SCORE})")
            for h in hits:
                ctype = h.get("chunk_type", "?")
                print(f"  [{ctype:8s}] score={h['score']:.3f} | {h['text'][:70]}...")

        context = self._build_context(hits) if hits else ''
        web_hint = ""
        if len(hits) < 2:
            web_hint = self.web_searcher.search_web_content(query)

        answer = self.llm.generate(query=query, context=context, web_hint=web_hint)

        return {
            'query': query,
            'answer': answer,
            'hits': len(hits),
            "sources": [h["text"][:120] for h in hits],
        }


if __name__ == '__main__':
    rag = MathRAG()
    print('ABC')
