import csv
from ast import List
from io import StringIO
from fastapi import APIRouter, File, UploadFile, HTTPException, status
import pandas as pd
from io import BytesIO
from app.service.review_service import analyze_review

from torch.package import analyze

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.post("/upload-pandas")
async def upload_csv_pandas(file: UploadFile = File(...)):
  content = await file.read()
  df = pd.read_csv(BytesIO(content))
  result = analyze_review(df)
  return result