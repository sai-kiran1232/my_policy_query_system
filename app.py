from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from rule_based_decision_system import process_query

app = FastAPI()

# Root route to test if the server is up
@app.get("/")
def home():
    return {"message": "Policy Query System is running 🚀"}

# API Key settings
API_KEY = "myhackrxkey2025"
api_key_header = APIKeyHeader(name="Authorization")

# Input model
class QueryInput(BaseModel):
    documents: str
    questions: list[str]

# Main API endpoint
@app.post("/hackrx/run")
async def run_model(
    payload: QueryInput,
    authorization: str = Depends(api_key_header)
):
    # Authorization check
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    answers = []

    # Process each question individually
    for question in payload.questions:
        try:
            answer = process_query(question)
            answers.append(answer)
        except Exception as e:
            print(f"Error processing question '{question}': {str(e)}")
            answers.append("An internal error occurred.")

    return {"answers": answers}
