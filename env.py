from dotenv import load_dotenv
load_dotenv()

import os

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
OPENAI_EMBEDDINGS_MODEL = os.environ['OPENAI_EMBEDDINGS_MODEL']
OPENAI_CHAT_MODEL = os.environ['OPENAI_CHAT_MODEL']

SERVER_HOST = os.environ['SERVER_HOST']
SERVER_PORT = int(os.environ['SERVER_PORT'])