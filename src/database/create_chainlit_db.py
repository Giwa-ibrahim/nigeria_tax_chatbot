"""
Initialize Chainlit database tables.
Run this once to create the necessary tables in PostgreSQL.
"""
import asyncio
import asyncpg
from src.configurations.config import settings

# SQL to create Chainlit tables
CREATE_TABLES_SQL = """
-- Create Thread table
CREATE TABLE IF NOT EXISTS "Thread" (
    id UUID PRIMARY KEY,
    "createdAt" TIMESTAMP,
    "updatedAt" TIMESTAMP,
    name TEXT,
    "userId" TEXT,
    "userIdentifier" TEXT,
    tags TEXT[],
    metadata JSONB
);

-- Create Step table
CREATE TABLE IF NOT EXISTS "Step" (
    id UUID PRIMARY KEY,
    name TEXT,
    type TEXT NOT NULL,
    "threadId" UUID REFERENCES "Thread"(id) ON DELETE CASCADE,
    "parentId" UUID,
    "disableFeedback" BOOLEAN DEFAULT FALSE,
    streaming BOOLEAN DEFAULT FALSE,
    "waitForAnswer" BOOLEAN DEFAULT FALSE,
    "isError" BOOLEAN DEFAULT FALSE,
    metadata JSONB,
    tags TEXT[],
    input TEXT,
    output TEXT,
    "createdAt" TIMESTAMP,
    "startTime" TIMESTAMP,
    "endTime" TIMESTAMP,
    generation JSONB,
    "showInput" TEXT,
    language TEXT,
    indent INTEGER
);

-- Create Element table
CREATE TABLE IF NOT EXISTS "Element" (
    id UUID PRIMARY KEY,
    "threadId" UUID REFERENCES "Thread"(id) ON DELETE CASCADE,
    type TEXT,
    url TEXT,
    "chainlitKey" TEXT,
    name TEXT NOT NULL,
    display TEXT,
    size TEXT,
    language TEXT,
    "forId" UUID,
    mime TEXT
);

-- Create User table
CREATE TABLE IF NOT EXISTS "User" (
    id UUID PRIMARY KEY,
    identifier TEXT NOT NULL UNIQUE,
    metadata JSONB,
    "createdAt" TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_thread_userId ON "Thread"("userId");
CREATE INDEX IF NOT EXISTS idx_step_threadId ON "Step"("threadId");
CREATE INDEX IF NOT EXISTS idx_element_threadId ON "Element"("threadId");
CREATE INDEX IF NOT EXISTS idx_user_identifier ON "User"(identifier);

"""

async def init_database():
    """Initialize Chainlit database tables."""
    db_url = settings.DATABASE_URL
    
    if not db_url:
        print("❌ DATABASE_URL not found in .env file")
        return
    
    try:
        print("🔗 Connecting to PostgreSQL...")
        conn = await asyncpg.connect(db_url)
        
        print("📝 Creating Chainlit tables...")
        await conn.execute(CREATE_TABLES_SQL)
        
        print("✅ Chainlit database tables created successfully!")
        print("\nTables created:")
        print("  - Thread")
        print("  - Step")
        print("  - Element")
        print("  - User")
        
        await conn.close()
        print("\n🎉 Database initialization complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(init_database())
