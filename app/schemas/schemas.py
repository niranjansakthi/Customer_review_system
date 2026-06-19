from pydantic import BaseModel, Field


class Review(BaseModel):
    review:str

class Response(BaseModel):
    sentiment: str
    emotion: str
    summary:str
