import sys
import asyncio

# Fix for Windows: Set event loop policy BEFORE any imports
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("✅ Windows event loop policy set to WindowsSelectorEventLoopPolicy")

# Configure logging to show module names
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Now import and run the app
import chainlit as cl
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentic_rag import main_agent


@cl.on_chat_start
async def start():
    """Welcome message when chat starts."""
    await cl.Message(
        content="👋 **Welcome to the Nigerian Tax Assistant!**\n\n"
                "I can help you with:\n"
                "- 📋 Tax policies and regulations\n"
                "- 💰 PAYE calculations\n"
                "- 🎯 Tax reliefs and exemptions\n\n"
                "Ask me anything about Nigerian taxation!"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages."""
    
    # Show thinking message
    msg = cl.Message(content="🤔 Thinking...")
    await msg.send()
    
    try:
        # Get user's question
        user_query = message.content
        
        # Use Chainlit's thread_id for conversation memory (persists across messages)
        thread_id = cl.context.session.thread_id
        
        # Call the main agent
        result = await main_agent(
            query=user_query,
            return_sources=False,
            thread_id=thread_id
        )
        
        # Format the response
        answer = result["answer"]
        
        # Build response
        response = f"{answer}"
        
        # Update message
        msg.content = response
        await msg.update()
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"❌ **Error:** {str(e)}\n\nPlease try again or rephrase your question."
        msg.content = error_msg
        await msg.update()
        print(f"Error details:\n{error_details}")
