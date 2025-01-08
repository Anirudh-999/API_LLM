from fastapi import FastAPI, HTTPException
from Gemini_LLM import ChatGemini
from langchain.schema.messages import HumanMessage
import pandas as pd

csv_data_path = r"C:\Users\Anirudh\Downloads\API_LLM (1)\API_LLM\synthetic_data_with_gst.csv"
try:
    data = pd.read_csv(csv_data_path)
    csv_data_str = data.to_string(index=False)  
except Exception as e:
    raise RuntimeError(f"Failed to load CSV data: {e}")

app = FastAPI(
    title="Gemini Server",
    version="1.0",
    description="API server using Gemini via LangChain.",
)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
gemini_model = ChatGemini(
    model_name="gemini-1.5-flash",
    credentials_path="C:\\Users\\SARTHAK\\Downloads\\gen-lang-client-0091686678-84db239ad662.json",
    generation_config=generation_config,
)

@app.post("/gemini")
async def invoke_gemini(input: dict):
    try:
        topic = input.get("topic", "").strip()

        if not topic:
            raise ValueError("Topic cannot be empty")
            
        topic_with_data = f"{topic}\n\nRelevant Data:\n{csv_data_str}"

        result = gemini_model.invoke([HumanMessage(content=topic_with_data)])

        response = {
            "prompt": topic_with_data,
            "response": result.get("response", ""),
            "status": result.get("status", "failed"),
        }

        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
