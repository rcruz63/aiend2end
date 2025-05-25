from fastapi import FastAPI
from pydantic import BaseModel
from agent.agent import agent
from agent.memory import load_messages, save_messages
from agent.setup_logging import setup_logging
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import os


# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")

app = FastAPI(title="Chatbot API", description="A simple chatbot API", version="1.0.0")

# Servir archivos estáticos (index.html)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Define a simple request model
class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None  # Optional conversation ID for tracking history


# Define a simple response model
class ChatResponse(BaseModel):
    response: str
    conversation_id: str


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(INDEX_HTML)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    chat_id = request.conversation_id or None
    # If no conversation_id is provided, start with an empty history
    if chat_id is None:
        previous_messages = []
    else:
        previous_messages = load_messages(request.conversation_id)
        logger.debug(f"Previous messages for conversation {chat_id}: {previous_messages}")

    response = agent(previous_messages + [{"role": "user", "content": request.message}])
    chat_id = save_messages(
        conversation_id=chat_id,
        messages=[
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": response}
        ]
    )
    logger.debug(f"Answer saved with conversation ID: {chat_id}: {response}")
    return ChatResponse(response=response, conversation_id=chat_id)


@app.get("/history/{conversation_id}")
async def history(conversation_id: str):
    return load_messages(conversation_id)
