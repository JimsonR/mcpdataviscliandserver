#!/usr/bin/env python3
"""
Test script for the StructuredAgent streaming endpoint
"""
import asyncio
import aiohttp
import json

async def test_structured_streaming():
    """Test the /llm/structured-agent-stream endpoint"""
    
    url = "http://localhost:8000/llm/structured-agent-stream"
    
    test_request = {
        "message": "What are the total sales for territory ID 1? And what's the weather like in New York?",
        "history": [],
        "chat_id": None
    }
    
    print("Testing structured agent streaming endpoint...")
    print(f"Request: {test_request['message']}")
    print("-" * 50)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=test_request) as response:
                print(f"Response status: {response.status}")
                print(f"Content type: {response.content_type}")
                print("-" * 50)
                
                if response.status == 200:
                    chunk_count = 0
                    async for chunk in response.content.iter_any():
                        if chunk:
                            chunk_count += 1
                            try:
                                # Each chunk should be a JSON line
                                chunk_str = chunk.decode('utf-8').strip()
                                if chunk_str:
                                    data = json.loads(chunk_str)
                                    print(f"Chunk {chunk_count}: {data}")
                            except json.JSONDecodeError:
                                print(f"Chunk {chunk_count} (raw): {chunk.decode('utf-8')}")
                    
                    print(f"\nTotal chunks received: {chunk_count}")
                else:
                    error_text = await response.text()
                    print(f"Error response: {error_text}")
                    
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_structured_streaming())
