import uvicorn
from dotenv import load_dotenv


def main():
    load_dotenv()
    uvicorn.run("agent.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
