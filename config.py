import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 256))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.6))
