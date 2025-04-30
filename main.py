from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from models import Models

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from ingest import ingest_files
from chat_ui import query_rag

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {'health_check': 'OK'}

@app.get("/info")
def info():
    return {'name': 'AI-Doc-Assistant', 'description': 'Querying API for uploaded documents'}

@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        answer = query_rag(request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# End-point to accept file uploads
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        temp_file_path = f"./data/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Ingest the file
        ingest_files(temp_file_path)

        # Optionally, you can remove the file after ingestion
        os.remove(temp_file_path)

        return {"message": f"File '{file.filename}' uploaded and ingested successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
