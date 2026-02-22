import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pymongo import MongoClient

app = FastAPI(title="AI Study Bot")

# 1. MongoDB Connection Setup (Direct Link)
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["study_bot_db"]
history_collection = db["chats"]

# 2. LLM Setup (Direct Key)
# NOTE: Agar ye key error de, toh console.groq.com se 'gsk_...' wali key lein.
api_key = os.getenv("API_KEY")
llm = ChatGroq(
    groq_api_key=api_key, 
    model_name="llama-3.3-70b-versatile"
)

class ChatInput(BaseModel):
    user_id: str
    message: str

@app.get("/")
def home():
    return {"message": "Study Bot API is Running!"}

@app.post("/chat")
async def chat_endpoint(chat_input: ChatInput):
    try:
        user_id = chat_input.user_id
        user_query = chat_input.message

        past_chats = list(history_collection.find({"user_id": user_id}).limit(5))
        
        messages = [
            SystemMessage(content="You are a helpful AI Study Assistant.")
        ]

        for chat in past_chats:
            messages.append(HumanMessage(content=chat["user_query"]))
            messages.append(AIMessage(content=chat["bot_response"]))

        messages.append(HumanMessage(content=user_query))

        response = llm.invoke(messages)
        bot_reply = response.content

        history_collection.insert_one({
            "user_id": user_id,
            "user_query": user_query,
            "bot_response": bot_reply
        })

        return {"user_id": user_id, "bot_response": bot_reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)