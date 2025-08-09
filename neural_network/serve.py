import os, json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your live server origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.get("/latest")
def latest():
    path = os.path.join("snapshots", "latest.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No snapshot yet")
    with open(path, "r") as f:
        data = json.load(f)
    # disable caches
    return JSONResponse(content=data, headers={"Cache-Control": "no-store"})
