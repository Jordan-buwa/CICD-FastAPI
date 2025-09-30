import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

LOGISTIC_MODEL = os.getenv("LOGISTIC_MODEL")
RF_MODEL = os.getenv("RF_MODEL")
