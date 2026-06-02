"""
Utility functions for WhatsApp webhook handling

This module contains helper functions for:
- Signature verification
- Message validation and extraction
- Sending messages to WhatsApp
- Media download
"""

import hmac
import hashlib
import logging
from typing import Dict, Any, Optional

import httpx

from src.configurations.config import settings
from .whatsapp_schema import WhatsAppMessage

logger = logging.getLogger("whatsapp_utils")


# ============================================================================
# Signature Verification
# ============================================================================

def verify_signature(payload: bytes, signature_header: str) -> bool:
    """
    Verify the webhook signature from WhatsApp.
    
    Args:
        payload: The raw request body
        signature_header: The X-Hub-Signature-256 header value
        
    Returns:
        True if signature is valid, False otherwise
    """
    if not signature_header:
        return False
    
    try:
        # Extract the signature (format: "sha256=<signature>")
        signature = signature_header.split("=")[1]
        
        # Calculate expected signature
        expected_signature = hmac.new(
            settings.APP_SECRET.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        return hmac.compare_digest(signature, expected_signature)
    except Exception as e:
        logger.error(f"Error verifying signature: {e}")
        return False


# ============================================================================
# Message Validation
# ============================================================================

def is_valid_whatsapp_message(data: Dict[str, Any]) -> bool:
    """
    Check if the webhook data contains a valid WhatsApp message.
    
    Args:
        data: The webhook payload
        
    Returns:
        True if it's a valid message, False otherwise
    """
    try:
        return (
            data.get("object") == "whatsapp_business_account" and
            "entry" in data and
            len(data["entry"]) > 0 and
            "changes" in data["entry"][0] and
            len(data["entry"][0]["changes"]) > 0 and
            "value" in data["entry"][0]["changes"][0] and
            "messages" in data["entry"][0]["changes"][0]["value"]
        )
    except Exception:
        return False


# ============================================================================
# Message Extraction
# ============================================================================

def extract_message_data(data: Dict[str, Any]) -> Optional[WhatsAppMessage]:
    """
    Extract message data from the webhook payload.
    
    Args:
        data: The webhook payload
        
    Returns:
        WhatsAppMessage object or None if extraction fails
    """
    try:
        # Navigate through the nested structure
        value = data["entry"][0]["changes"][0]["value"]
        messages = value.get("messages", [])
        
        if not messages:
            return None
        
        message = messages[0]  # Get the first message
        
        # Extract message details
        message_type = message.get("type", "unknown")
        from_number = message.get("from", "")
        message_id = message.get("id", "")
        timestamp = message.get("timestamp", "")
        
        # Extract text content
        text_body = None
        media_url = None
        media_id = None
        
        if message_type == "text":
            text_body = message.get("text", {}).get("body", "")
        elif message_type == "image":
            media_id = message.get("image", {}).get("id", "")
        elif message_type == "audio":
            media_id = message.get("audio", {}).get("id", "")
        elif message_type == "video":
            media_id = message.get("video", {}).get("id", "")
        elif message_type == "document":
            media_id = message.get("document", {}).get("id", "")
        
        return WhatsAppMessage(
            from_number=from_number,
            message_id=message_id,
            timestamp=timestamp,
            message_type=message_type,
            text_body=text_body,
            media_url=media_url,
            media_id=media_id
        )
        
    except Exception as e:
        logger.error(f"Error extracting message data: {e}", exc_info=True)
        return None


# ============================================================================
# Send Message to WhatsApp
# ============================================================================

async def send_whatsapp_message(to_number: str, message_text: str) -> Dict[str, Any]:
    """
    Send a WhatsApp message using the WhatsApp Business API.
    
    Args:
        to_number: Recipient's phone number (without + prefix)
        message_text: Message text to send
        
    Returns:
        API response dictionary
        
    Raises:
        httpx.HTTPStatusError: If the API request fails
    """
    url = f"https://graph.facebook.com/{settings.VERSION}/{settings.PHONE_NUMBER_ID}/messages"
    
    # Format phone number
    #formatted_number = format_phone_number(to_number)
    
    headers = {
        "Authorization": f"Bearer {settings.ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number, 
        "type": "text",
        "text": {
            "preview_url": False,
            "body": message_text
        }
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            logger.info(f"✅ Message sent successfully to {to_number}")
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ HTTP error sending message: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"❌ Error sending message: {e}")
        raise


# ============================================================================
# Download WhatsApp Media
# ============================================================================

async def download_whatsapp_media(media_id: str) -> bytes:
    """
    Download media from WhatsApp.
    
    Args:
        media_id: The media ID from WhatsApp
        
    Returns:
        The media file as bytes
        
    Raises:
        httpx.HTTPStatusError: If the API request fails
    """
    # Step 1: Get media URL
    url = f"https://graph.facebook.com/{settings.VERSION}/{media_id}"
    headers = {"Authorization": f"Bearer {settings.ACCESS_TOKEN}"}
    
    try:
        async with httpx.AsyncClient() as client:
            # Get media info
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            media_info = response.json()
            
            # Step 2: Download media
            media_url = media_info.get("url")
            if not media_url:
                raise ValueError("No media URL found in response")
            
            response = await client.get(media_url, headers=headers)
            response.raise_for_status()
            
            logger.info(f"✅ Downloaded media {media_id} successfully")
            return response.content
            
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ HTTP error downloading media: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"❌ Error downloading media: {e}")
        raise


# ============================================================================
# Format Phone Number
# ============================================================================

def format_phone_number(phone: str) -> str:
    """
    Format phone number to E.164 format.
    
    Args:
        phone: Phone number in any format
        
    Returns:
        Phone number in E.164 format (e.g., 2348173215185)
    """
    # Remove all non-digit characters
    digits = ''.join(filter(str.isdigit, phone))
    
    # Nigerian numbers
    if digits.startswith("0") and len(digits) == 11:
        return "234" + digits[1:]

    if digits.startswith("234"):
        return digits
    
    return digits


# ============================================================================
# Validate Message Length
# ============================================================================

def truncate_message(message: str, max_length: int = 4096) -> str:
    """
    Truncate message to WhatsApp's maximum length.
    
    WhatsApp has a 4096 character limit for text messages.
    
    Args:
        message: The message text
        max_length: Maximum allowed length (default: 4096)
        
    Returns:
        Truncated message if necessary
    """
    if len(message) <= max_length:
        return message
    
    # Truncate and add ellipsis
    truncated = message[:max_length - 3] + "..."
    logger.warning(f"Message truncated from {len(message)} to {len(truncated)} characters")
    
    return truncated
