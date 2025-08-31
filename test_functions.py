import asyncio
import json
from main import handle_text_query

async def test_queries():
    test_cases = [
        "What's 15 * 8 + 3?",
        "Find me papers about machine learning",
        "Hello, how are you doing today?",
        "Calculate the square root of 144",
        "Search for quantum physics research"
    ]
    
    for query in test_cases:
        print(f"\\n=== Testing: {query} ===")
        
        try:
            response = await handle_text_query(query, "test_session")
            
            print(f"User Query: {query}")
            print(f"Is Function Call: {response['is_function_call']}")
            
            if response['is_function_call']:
                print(f"Function Called: {json.dumps(response['function_call'], indent=2)}")
            
            print(f"Final Response: {response['text_response']}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error testing query '{query}': {e}")

if __name__ == "__main__":
    asyncio.run(test_queries())