from rag import MathRAG

def main():
    rag = MathRAG()
    print('-----------------------MATH RAG-------------------')
    print('Nhập câu hỏi toán học: (gõ exit để thoát)')

    while True:
        query = input('\n[Câu hỏi] ').strip()
        if not query or query.lower() in ("exit", "quit", "q"):
            break

        result = rag.answer(query, verbose=True)
        print(f"\n[Trả lời]\n{result['answer']}")

if __name__ == '__main__':
    main()