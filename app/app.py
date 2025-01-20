from fastapi import FastAPI

from routers import detection

app = FastAPI()

app.include_router(detection.router)
