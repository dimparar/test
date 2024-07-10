from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

embed_router = APIRouter()


class EmbedRequest(BaseModel):
    embedding_model: str = Field(..., description="The name of the embedding model to use")
    texts: List[str] = Field(..., description="List of texts to generate embeddings for")


# Dictionary of supported embedding models.
EMBEDDING_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {"dimensions": 384},
}

# Cache for initialized embedding models.
initialized_models = {}


def get_embedding_model(model_name: str):
    if model_name not in initialized_models:
        if model_name.startswith("sentence-transformers/"):
            initialized_models[model_name] = SentenceTransformer(model_name)

    return initialized_models[model_name]


@embed_router.post("/")
async def generate_embeddings(request: EmbedRequest):
    """
    Generate embeddings for a list of texts using the specified embedding model.

    This endpoint takes a list of texts and an embedding model name, and returns
    the generated embeddings along with metadata about the operation.

    Args:
        request (EmbedRequest): The request body containing the embedding model and texts.

    Returns:
        JSONResponse: A JSON response containing the generated embeddings and metadata.

    Raises:
        HTTPException: If the requested embedding model is not supported.

    Example:
    Request Body:
    {
        "embedding_model": "model_name",
        "texts": ["text1", "text2", "text3"]
    }

    Response:
    {
        "data": {
            "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
        },
        "meta": {
            "model": "model_name",
            "dimensions": 768,
            "count": 3
        }
    }
    """
    embedding_model = request.embedding_model
    texts = request.texts

    # Check if the requested embedding model is supported
    if embedding_model not in EMBEDDING_MODELS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported embedding model: {embedding_model}"
        )

    model_dimensions = EMBEDDING_MODELS[embedding_model]["dimensions"]

    embedding_model_instance = get_embedding_model(embedding_model)

    embeddings_list = embedding_model_instance.encode(texts).tolist()

    response_data = {
        "data": {
            "embeddings": embeddings_list,
        },
        "meta": {
            "model": embedding_model,
            "dimensions": model_dimensions,
            "count": len(embeddings_list),
        }
    }

    return JSONResponse(
        content=response_data,
        status_code=200,
        headers={"X-API-Version": "1.0"}
    )
