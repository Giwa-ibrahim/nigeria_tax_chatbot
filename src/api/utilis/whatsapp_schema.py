"""
Models for WhatsApp webhook messages and responses.
"""

from pydantic import BaseModel, Field


class WhatsAppMessage(BaseModel):
    """Model for incoming WhatsApp message"""
    from_number: str = Field(..., description="Sender's phone number")
    message_id: str = Field(..., description="Unique message ID")
    timestamp: str = Field(..., description="Message timestamp")
    message_type: str = Field(..., description="Type of message (text, image, audio, etc.)")
    text_body: str | None = Field(None, description="Text content of the message")
    media_url: str | None = Field(None, description="URL for media messages")
    media_id: str | None = Field(None, description="Media ID for media messages")


class WebhookResponse(BaseModel):
    """Response model for webhook"""
    status: str
    message: str
