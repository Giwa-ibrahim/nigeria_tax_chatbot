"""
Main App Tables Schema Inspector.
Queries the 5 main app tables to show available columns for personalization.

Usage:
    python -m inspect_table
"""
import asyncio
import logging
from sqlalchemy import text
from src.database.connection import get_db_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger("table_inspector")


async def inspect_table(table_name: str):
    """
    Inspect a table and show columns with sample data.
    """
    try:
        async with get_db_session() as session:
            # Get column information
            query = text(f"SELECT * FROM {table_name} LIMIT 1")
            result = await session.execute(query)
            
            # Check if table has data
            row = result.fetchone()
            
            if row:
                # Get column names
                columns = result.keys()
                
                print(f"\n{'='*80}")
                print(f"📋 TABLE: {table_name}")
                print(f"{'='*80}")
                print(f"\n✅ COLUMNS ({len(columns)} fields):")
                
                # Convert row to dict for easier access
                row_dict = dict(row._mapping)
                
                # Print columns with their sample values
                for i, col in enumerate(columns, 1):
                    value = row_dict[col]
                    value_type = type(value).__name__
                    
                    # Truncate long values
                    if isinstance(value, str) and len(value) > 50:
                        value_display = f"{value[:50]}..."
                    else:
                        value_display = value
                    
                    print(f"  {i:2d}. {col:30s} | Type: {value_type:15s} | Sample: {value_display}")
                
                print(f"\n{'='*80}\n")
                
            else:
                print(f"\n⚠️  TABLE '{table_name}' EXISTS but is EMPTY (no rows)")
                
                # Try to get column names from schema
                schema_query = text(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position
                """)
                schema_result = await session.execute(schema_query)
                schema_rows = schema_result.fetchall()
                
                if schema_rows:
                    print(f"\n📋 COLUMNS ({len(schema_rows)} fields):")
                    for i, (col_name, data_type) in enumerate(schema_rows, 1):
                        print(f"  {i:2d}. {col_name:30s} | Type: {data_type}")
                    print(f"\n{'='*80}\n")
                
    except Exception as e:
        print(f"\n❌ ERROR querying '{table_name}': {e}\n")


async def main():
    """
    Inspect all 5 main app tables.
    """
    print("\n" + "="*80)
    print("🔍 MAIN APP TABLES SCHEMA INSPECTION")
    print("="*80)
    print("\nQuerying 5 tables to identify fields for chatbot personalization...")
    
    tables = [
        "profiles",
        "financial_profiles", 
        "financial_income",
        "financial_expenses",
        "tax_calculations"
    ]
    
    for table in tables:
        await inspect_table(table)
    
    print("\n" + "="*80)
    print("✅ INSPECTION COMPLETE")
    print("="*80)
    print("\n💡 USE THIS DATA TO:")
    print("  1. Identify which fields exist in main app")
    print("  2. Map fields to chatbot session context")
    print("  3. Build SQL queries for user data enrichment")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())