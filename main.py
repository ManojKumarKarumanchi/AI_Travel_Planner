""" FastAPI Main Template"""

import os
import uvicorn
from fastapi import FastAPI

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Initialize the FastAPI app application
app = FastAPI(title=os.getenv("APP_NAME"),
            description=os.getenv("APP_DESCRIPTION"),
            version=os.getenv("APP_VERSION"))

@app.get("/")
async def read_root():
    """Basic asynchronous GET endpoint."""
    return {"message": "Welcome to AI Travel Planner"}

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app,
                host=os.getenv("HOST"),
                port=int(os.getenv("PORT")),
                log_level="info",
                reload=True)
