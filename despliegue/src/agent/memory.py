import os
# from datetime import datetime, timezone
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
from agent.setup_logging import setup_logging
import logging

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

url: str = os.environ.get("SUPABASE_URL") or ""
key: str = os.environ.get("SUPABASE_KEY") or ""

supabase: Client = create_client(url, key)


def check_connection() -> bool:
    """
    Checks if the Supabase connection is working.
    """
    try:
        # Try a simple select on a known table
        supabase.table("messages").select("id").limit(1).execute()
        return True
    except Exception as e:
        logger.error(f"Supabase connection error: {e}")
        print(f"Supabase connection error: {e}")
        return False


def load_messages(conversation_id: Optional[str]) -> List[Dict[str, Any]]:
    """
    Load all messages for a given conversation_id (chatId).
    """
    if not check_connection() or not conversation_id:
        return []
    try:
        response = (
            supabase.table("messages")
            .select("*")
            .eq("chatId", conversation_id)
            .order("createdAt", desc=False)
            .execute()
        )
        messages = []
        for msg in response.data:
            messages.append(msg["content"])
        logger.debug(f"Loaded messages for conversation {conversation_id}: {messages}")
        return messages
    except Exception as e:
        logger.error(f"Error loading messages: {e}")
        print(f"Error loading messages: {e}")
        return []


def save_messages(
    conversation_id: Optional[str],
    messages: List[Dict[str, Any]],
    user_id: Optional[str] = None,
) -> str:
    """
    Save a list of messages to the database for a given conversation_id.
    If conversation_id is None, creates a new chat and returns its id.
    Each message should be a dict with at least: role, content, createdAt.
    """
    if not check_connection() or not messages:
        logger.error("No connection to Supabase or no messages to save.")
        raise ValueError("No connection to Supabase or no messages to save.")

    chat_id = conversation_id
    try:
        # Si es una nueva conversación, crea el registro en chat
        if chat_id is None:
            first_message = messages[0]
            titulo = first_message.get("content", "")[:100]  # Primeros 100 caracteres
            chat_data = {
                # "id": chat_id,
                "title": titulo,
                "visibility": "private",
            }
            response = supabase.table("chat").insert(chat_data).execute()
            if response.data and len(response.data) > 0:
                chat_id = response.data[0]["id"]
            else:
                logger.error("No se pudo obtener el id del chat insertado")
                print("No se pudo obtener el id del chat insertado")
                raise ValueError("No se pudo obtener el id del chat insertado")
            logger.debug(f"Respuesta Insert chat: {response}")

        # Prepara los mensajes para insertar
        for msg in messages:
            to_insert = {
                "content": msg,
                "chatId": chat_id,
            }
            response = supabase.table("messages").insert(to_insert).execute()
            logger.debug(f"Respuesta Insert message: {response}")
        return chat_id
    except Exception as e:
        logger.error(f"Error saving messages: {e}")
        print(f"Error saving messages: {e}")
        raise ValueError(f"Error saving messages: {e}")
