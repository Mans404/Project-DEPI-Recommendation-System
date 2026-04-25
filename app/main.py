from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routers import recommend
from app.model import load_resources

app = FastAPI(
    title="Movie Recommender API",
    description="KNN-based content movie recommender",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    load_resources()

# Serve the UI at the root URL
@app.get("/", response_class=FileResponse)
def serve_ui():
    return "static/movies_ui.html"

# Mount static folder for any extra assets (CSS, JS, images) in the future
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(recommend.router, prefix="/api")