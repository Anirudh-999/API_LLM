# from langchain.schema import LLMResult
# from langchain.schema.messages import AIMessage, HumanMessage
# from langchain.schema import ChatResult
import google.generativeai as genai
from langchain_core.runnables import Runnable
import os


class ChatGemini(Runnable):
    def __init__(self, model_name: str, credentials_path: str, generation_config: dict):

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        self.model = genai.GenerativeModel(
            model_name=model_name, 
            generation_config=generation_config
        )

    def invoke(self, input_data: str):
        print("this is llm invoke")
        print(input_data)
        prompt = input_data 

        chat_session = self.model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        response_text = str(response.candidates[0].content).strip() 
        return response_text