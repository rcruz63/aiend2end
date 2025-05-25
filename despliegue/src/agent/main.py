import uvicorn
from dotenv import load_dotenv
from agent.setup_logging import setup_logging
import logging

# Configurar logging
setup_logging()


def main():
    load_dotenv()
    uvicorn.run("agent.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
