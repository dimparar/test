from fastapi import FastAPI
from api import embed
from contextlib import asynccontextmanager
import logging
import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.info("Starting application")
    yield

    # Shutdown code
    # Perform any necessary cleanup tasks
    # e.g., close database connections, release resources
    # ...
    logging.info("Shutting down application")


app = FastAPI(
    title="Embedding Service",
    description="A service to facilitate embedding needs for Falcon AI",
    version="0.1.0",
    lifespan=lifespan,
)

# Include API routes
app.include_router(embed.embed_router, prefix="/api/v1/embed", tags=["Embedding"])


@app.get("/")
async def root():
    return {"message": "Falcon Embedding Service Running."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.APP_HOST, port=config.APP_PORT)
