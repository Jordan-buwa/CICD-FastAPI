from config import API_KEY
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def require_api_key(api_key: str = Security(api_key_header)):
    print(api_key)  # what user is sending
    print(API_KEY)  # what we are expecting to see
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
