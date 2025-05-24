from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Chatbot API", description="A simple chatbot API", version="1.0.0")


# Define a simple request model
class ChatRequest(BaseModel):
    message: str


# Define a simple response model
class ChatResponse(BaseModel):
    response: str


@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Here you would typically process the request and generate a response
    # For demonstration, we will just echo the message back
    response_message = "Hola, soy un chatbot genérico. ¿En qué puedo ayudarte hoy? tu mensaje fue: " + request.message
    return ChatResponse(response=response_message)
