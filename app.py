from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from rule_based_decision_system import process_query

app = FastAPI()

API_KEY = "myhackrxkey2025"
api_key_header = APIKeyHeader(name="Authorization")

class QueryInput(BaseModel):
    documents: str
    questions: list[str]

@app.post("/hackrx/run")
async def run_model(
    payload: QueryInput,
    authorization: str = Depends(api_key_header)
):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    answers = []
    for question in payload.questions:
        try:
            answer = process_query(question)
            answers.append(answer)
        except Exception as e:
            print("Error:", e)
            answers.append("An internal error occurred.")

    return {"answers": answers}
