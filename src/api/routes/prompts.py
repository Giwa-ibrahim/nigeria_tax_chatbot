"""
Personalized Prompts API Endpoint

Simple endpoint returning 8 example prompts.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from typing import List
from pydantic import BaseModel

from src.services.personalized_prompts import get_personalized_prompts

logger = logging.getLogger("prompts_api")

router = APIRouter(prefix="/api/v1/prompts", tags=["prompts"])


class PromptsResponse(BaseModel):
    """Simple response with prompts and user_id."""
    prompts: List[str]
    user_id: str


@router.get("/personalised_prompts", response_model=PromptsResponse)
async def get_prompts(user_id: str) -> PromptsResponse:
    """
    Get 8 personalized prompts for querying the chatbot.
    
    **Query Parameters:**
    - user_id: User identifier (required)
    
    **Returns:**
    - List of 8 example questions covering tax, PAYE, financial, and combined topics
    """
    try:
        logger.info(f"📝 Fetching prompts for user: {user_id}")
        
        prompts = get_personalized_prompts()
        
        return PromptsResponse(
            prompts=prompts,
            user_id=user_id
        )
        
    except Exception as e:
        logger.error(f"❌ Error fetching prompts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching prompts: {str(e)}"
        )
