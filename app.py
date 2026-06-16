import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from rag import MathRAG


app = FastAPI(title="Meta Math Assistant")
app.mount("/static", StaticFiles(directory="frontend"), name="static")

chatbot = MathRAG()
sessions_memory = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = 'default_user'

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    hits: int
    sources: list[str]

@app.get('/')
def read_root():
    return FileResponse("frontend/index.html")

@app.post('/chat', response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message không được để trống")

    history = sessions_memory.get(request.session_id, [])
    try:
        result = chatbot.answer(query=request.message, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    history.append(HumanMessage(content=request.message))
    history.append(AIMessage(content=result['answer']))
    sessions_memory[request.session_id] = history[-20:]

    return ChatResponse(
        answer=result.get('answer', 'Không có câu trả lời từ hệ thống.'),
        session_id=request.session_id,
        hits=result.get('hits', 0),
        sources=result.get('sources', [])
    )

@app.get('/history/{session_id}')
def get_history(session_id: str):
    history = sessions_memory.get(session_id, [])
    return {
        'session_id': session_id,
        'messages': [
            {'role': 'user' if isinstance(m, HumanMessage) else 'assistant', 'content': m.content}
            for m in history
        ]
    }

@app.delete('/history/{session_id}')
def clear_history(session_id: str):
    sessions_memory.pop(session_id, None)
    return {'message': 'Đã xóa lịch sử của {}'.format(session_id)}


@app.get('/health')
def health():
    return {'status': 'ok', 'sessions': len(sessions_memory)}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)