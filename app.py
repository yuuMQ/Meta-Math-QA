from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Meta Math Assistant")




if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.0', port=8000)