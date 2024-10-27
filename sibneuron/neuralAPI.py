import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from starlette.responses import RedirectResponse
from urllib3 import request

from net import load_data, InstructionMemory, find_similar_topics_and_solutions

keras = tf.keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

app = FastAPI()

model = keras.models.load_model('my_model.keras')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)


class Topic(BaseModel):
    text: str


@app.post("/predict")
async def predict_topic(topic: Topic):
        global instruction_memory
        instruction_memory = InstructionMemory()
        instruction_memory.load_docx_files(os.getcwd())
        print("Инициализация завершена")

        topic = topic.text

        print(topic)
        seq = tokenizer.texts_to_sequences([topic])
        pad = keras.preprocessing.sequence.pad_sequences(seq)

        predicted_label = model.predict(pad)
        predicted_class = label_encoder.inverse_transform([np.argmax(predicted_label)])

        data, _, _ = load_data()
        solution = data.loc[data['Topic'] == topic, 'Solution'].values
        solution_text = solution[0] if len(solution) > 0 else "Решение не найдено."

        instruction_file = instruction_memory.find_instruction_file(topic)
        instruction_file_info = instruction_file[0] if instruction_file else ("Не найдено", "", 0)
        fileinf = instruction_file_info[0]

        similar_topics = find_similar_topics_and_solutions(topic, data, instruction_memory)

        url = "http://127.0.0.1:8080/topic"

        simtopics = ''
        for index, topic, soln, similarity in similar_topics:
            if index is not None:
                simtopics += f"Номер строки: {index + 1}, Топик: {str(topic)}, Решение: {str(soln)}"
                simtopics += "\n"
            else:
                simtopics += f"- {topic}, {soln}"
                simtopics += "\n"

        answer = "Topic: " + predicted_class[0] + "\n" + "Solution text: " + solution_text + "\n" + "Filename: " + fileinf + "\n" + simtopics

        async with httpx.AsyncClient() as client:
            response = await client.post(url,json= {"text": answer})
            print(response.status_code, response.text)
        return RedirectResponse(url="/", status_code=301)



@app.get("/")
async def root():
    return {"message": "Welcome to the Topic Prediction API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8883)