import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_PATH = os.getenv('MODEL_PATH', 'openlm-research/open_llama_7b')
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 32))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
