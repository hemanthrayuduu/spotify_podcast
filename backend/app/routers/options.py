"""Form vocabulary endpoint: the frontend renders its form from this."""

from fastapi import APIRouter

from app.core.vocabulary import options_payload

router = APIRouter()


@router.get("/options")
async def options():
    return options_payload()
