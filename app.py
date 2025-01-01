from fastapi import FastAPI, Request, HTTPException
from langserve import add_routes
from gemini_llm import ChatGemini 

app = FastAPI(
    title="Gemini Server",
    version="1.0",
    description="API server using Gemini via LangChain.",
)

generation_config = {
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}
gemini_model = ChatGemini(
    model_name="gemini-1.5-flash",
    credentials_path=r"C:\Users\Anirudh\Downloads\gen-lang-client-0503948327-7a4a4cd77c34.json",
    generation_config=generation_config,
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Gemini Server"}

@app.post("/invoke")
async def invoke_model(request: Request):
    try:
        data = await request.json()  
        if "input" not in data:
            raise ValueError("Missing 'input' in request payload")
        result = gemini_model.invoke(data['input']['topic']) 
        print(result)
        return {"output": result}

    except ValueError as ve:
        return HTTPException(status_code=400, detail=f"Invalid request: {ve}")
    except Exception as e:
        print(f"Internal Server Error: {e}") 
        return HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)