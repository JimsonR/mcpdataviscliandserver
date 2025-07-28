import redis
import json
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
# REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

def get_chat_history(chat_id):
    data = redis_client.get(f"chat:{chat_id}")
    return json.loads(data) if data else []

def save_chat_history(chat_id, history):
    redis_client.set(f"chat:{chat_id}", json.dumps(history))

def delete_chat_session(chat_id):
    redis_client.delete(f"chat:{chat_id}")

def list_chat_sessions():
    keys = redis_client.keys("chat:*")
    return [key.split("chat:")[1] for key in keys]