# server.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from Wiki.chatbot import WikiChatbot

app = FastAPI(title="Wikipedia + Ollama Chatbot (sync with Langfuse)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# instantiate chatbot once (will initialize Langfuse & Ollama)
bot = WikiChatbot()

@app.post("/chat")
async def chat(request: Request):
    payload = await request.json()
    question = payload.get("question", "").strip()
    if not question:
        return {"error": "question required"}

    # answer_stream is a generator that yields tokens (synchronous generator)
    def stream():
        for token in bot.answer_stream(question):
            # token might be bytes or str; ensure bytes
            if isinstance(token, str):
                yield token.encode("utf-8")
            else:
                yield token

    return StreamingResponse(stream(), media_type="text/plain")
