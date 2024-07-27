from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Serve the static HTML file
@app.get("/", response_class=HTMLResponse)
async def read_html():
    # Path to your HTML file
    file_path = os.path.join("templates", "index.html")
    return FileResponse(file_path)
#uvicorn app:app --reload
