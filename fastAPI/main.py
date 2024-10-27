import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from flask import jsonify
from pydantic import BaseModel
from starlette.responses import RedirectResponse
from typing import List

app = FastAPI()

@app.get("/")
async def root():
    return
@app.get("/ping")
async def ping():
    return JSONResponse(content = {})
@app.get("/topic/{topic}")
async def get_topic(topic:str):
    print(topic)
    url = "http://127.0.0.1:8883/predict"  # URL вашего другого микросервиса
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json={"text": topic})
    return RedirectResponse(url="/ping")

class Topic(BaseModel):
    text: str

@app.post("/topic")
async def send_answer(topic: Topic):
    print(topic)
    url = "http://127.0.0.1:5000/"
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json= {"text": topic.text})
    return RedirectResponse(url="/ping")

if __name__ == "__main__":
    uvicorn.run(app,host = "127.0.0.1",port = 8080)