"""
WhatsApp Webhook API for Nigerian Tax Chatbot

This module handles WhatsApp webhook verification and incoming messages.
It integrates with the Meta WhatsApp Business API.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Request, Response, HTTPException, Query, status

from src.configurations.config import settings
from src.agent.main_agent import main_agent
from .. import WhatsAppMessage, WebhookResponse
from .. import (
    verify_signature,
    is_valid_whatsapp_message,
    extract_message_data,
    send_whatsapp_message,
    truncate_message
)

# Configure logging
logger = logging.getLogger('whatsapp_webhook')

# Create router
router = APIRouter(
    prefix="/webhook",
    tags=["WhatsApp Webhook"]
)


# ============================================================================
# Webhook Verification (GET)
# ============================================================================

@router.get("", summary="Verify WhatsApp Webhook")
async def verify_webhook(
    request: Request,
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    """
    Webhook verification endpoint for WhatsApp.
    
    WhatsApp will send a GET request with the following parameters:
    - hub.mode: Should be 'subscribe'
    - hub.verify_token: Your verification token (should match ENDPOINT_AUTH_KEY)
    - hub.challenge: Random string to echo back
    
    Returns:
        The hub.challenge value if verification succeeds
    """
    logger.info(f"Webhook verification request received")
    logger.info(f"Mode: {hub_mode}, Token: {hub_verify_token}, Challenge: {hub_challenge}")
    
    # Verify the request
    if hub_mode == "subscribe" and hub_verify_token == settings.ENDPOINT_AUTH_KEY:
        logger.info("✅ Webhook verified successfully")
        return Response(content=hub_challenge, media_type="text/plain")
    else:
        logger.warning("❌ Webhook verification failed")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Verification failed"
        )


# ============================================================================
# Webhook Message Handler (POST)
# ============================================================================

@router.post("", summary="Handle Incoming WhatsApp Messages", response_model=WebhookResponse)
async def handle_webhook(request: Request) -> WebhookResponse:
    """
    Handle incoming WhatsApp messages.
    
    This endpoint receives POST requests from WhatsApp when users send messages.
    It processes the message and can trigger appropriate responses.
    
    Args:
        request: The incoming webhook request
        
    Returns:
        WebhookResponse confirming receipt
    """
    try:
        # Get the raw body for signature verification
        body = await request.body()
        
        # Check if body is empty
        if not body:
            logger.warning("⚠️ Received empty request body")
            return WebhookResponse(status="error", message="Empty request body")
        
        # Verify webhook signature (optional but recommended for production)
        signature = request.headers.get("X-Hub-Signature-256", "")
        if not verify_signature(body, signature):
            #logger.warning("⚠️ Invalid webhook signature")
            # In production, you might want to reject invalid signatures
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature"
            )
        
        # Parse the JSON payload
        try:
            data = await request.json()
            logger.info(f"📩 Received webhook data: {data}")
        except Exception as json_error:
            logger.error(f"❌ Invalid JSON in request body: {json_error}")
            return WebhookResponse(status="error", message="Invalid JSON format")
        
        # Process the webhook data
        if is_valid_whatsapp_message(data):
            message_data = extract_message_data(data)
            
            if message_data:
                logger.info(f"📱 Processing message from: {message_data.from_number}")
                logger.info(f"💬 Message type: {message_data.message_type}")
                logger.info(f"📝 Message content: {message_data.text_body}")
                
                # Process the message
                await process_whatsapp_message(message_data)
            else:
                logger.info("ℹ️ No processable message found in webhook data")
        else:
            logger.info("ℹ️ Received webhook notification (not a user message)")
        
        # Always return 200 OK to acknowledge receipt
        return WebhookResponse(status="success", message="Webhook received")
        
    except Exception as e:
        logger.error(f"❌ Error processing webhook: {str(e)}", exc_info=True)
        # Still return 200 to prevent WhatsApp from retrying
        return WebhookResponse(status="error", message=str(e))


# ============================================================================
# Message Processing
# ============================================================================

async def process_whatsapp_message(message: WhatsAppMessage):
    """
    Process the incoming WhatsApp message.
    
    This integrates with the main_agent to provide tax assistance.
    Uses phone number as both user_id (for tracking) and thread_id (for memory).
    
    Args:
        message: The extracted WhatsApp message
    """
    try:
        # Handle different message types
        if message.message_type == "text" and message.text_body:
            logger.info(f"Processing text message: {message.text_body}")
            
            phone_number = message.from_number
            
            try:
                from src.database.chat_manager import ChatManager
                
                # Track user activity
                await ChatManager.track_user_activity(phone_number)
                
                # Save user message
                try:
                    await ChatManager.add_user_message(
                        thread_id=phone_number,
                        content=message.text_body
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save user message: {e}")

                # Call the main agent
                result = await main_agent(
                    user_id=phone_number,      # Track the WhatsApp user
                    query=message.text_body,   # The user's question
                    return_sources=False,      # Don't include sources in WhatsApp response
                    thread_id=phone_number     # Use same phone for conversation memory
                )
                
                # Extract the answer
                response_text = result.get("answer")
                
                # Save assistant response
                try:
                    await ChatManager.add_assistant_message(
                        thread_id=phone_number,
                        content=response_text,
                        agent_type=result.get("route_used", "unknown"),
                        tokens_used=0
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save assistant message: {e}")
                
                # Truncate if needed (WhatsApp has 4096 char limit)
                response_text = truncate_message(response_text)
                
                # Send response back to WhatsApp
                await send_whatsapp_message(
                    to_number=phone_number,
                    message_text=response_text
                )
                
                logger.info(f"✅ Sent response to {phone_number} via {result.get('route_used', 'unknown')} route")
                
            except Exception as agent_error:
                logger.error(f"❌ Error running agent: {agent_error}", exc_info=True)
                # Send error message to user
                await send_whatsapp_message(
                    to_number=phone_number,
                    message_text="Sorry, I encountered an error processing your tax question. Please try again or rephrase your question."
                )
            
        elif message.message_type in ["image", "audio", "video", "document"]:
            logger.info(f"Received {message.message_type} message with ID: {message.media_id}")
            # Handle media messages
            # You might want to download and process the media
            # from src.api.routes.utils import download_whatsapp_media
            # media_data = await download_whatsapp_media(message.media_id)
            
            # For now, send a message that media is not supported
            await send_whatsapp_message(
                to_number=message.from_number,
                message_text="I received your file. Currently, I can only process text messages. Please send your question as text."
            )
            
        else:
            logger.info(f"Unsupported message type: {message.message_type}")
            await send_whatsapp_message(
                to_number=message.from_number,
                message_text="Sorry, I don't support this message type. Please send your question as text."
            )
            
    except Exception as e:
        logger.error(f"❌ Error processing message: {e}", exc_info=True)
        # Send error message to user
        try:
            await send_whatsapp_message(
                to_number=message.from_number,
                message_text="Sorry, I encountered an error processing your message. Please try again."
            )
        except Exception as send_error:
            logger.error(f"Failed to send error message: {send_error}")
