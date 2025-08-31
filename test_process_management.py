#!/usr/bin/env python3
"""
Test script to verify process management improvements.
"""
import requests
import json
import time

BASE_URL = "http://localhost:8080"

def test_process_status():
    """Test the process status endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/mcp/process-status")
        print("Process Status:")
        print(json.dumps(response.json(), indent=2))
        return response.json()
    except Exception as e:
        print(f"Error checking process status: {e}")
        return None

def test_multiple_requests():
    """Test multiple requests to see if processes accumulate."""
    print("Testing multiple requests...")
    
    # Check initial state
    initial_status = test_process_status()
    if initial_status:
        initial_processes = initial_status.get("active_stdio_processes", 0)
        print(f"Initial processes: {initial_processes}")
    
    # Make multiple requests
    for i in range(3):
        print(f"\nRequest {i+1}:")
        try:
            response = requests.post(
                f"{BASE_URL}/llm/structured-agent-stream",
                headers={"Content-Type": "application/json"},
                json={
                    "message": "Hello, test message",
                    "chat_id": f"test_session_{i}",
                    "history": []
                },
                stream=True,
                timeout=10
            )
            
            # Read just a bit of the response
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    data = json.loads(line)
                    print(f"Response type: {data.get('type')}")
                    if data.get('type') in ['final_answer', 'error']:
                        break
                    if data.get('type') == 'tool_use':
                        print(f"Tool used: {data.get('tool_name')} from server: {data.get('server')}")
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
        
        # Check process count after each request
        status = test_process_status()
        if status:
            current_processes = status.get("active_stdio_processes", 0)
            print(f"Processes after request {i+1}: {current_processes}")
        
        time.sleep(1)
    
    # Final check
    print("\nFinal process status:")
    test_process_status()

def test_cleanup():
    """Test manual cleanup."""
    print("\nTesting manual cleanup...")
    try:
        response = requests.post(f"{BASE_URL}/mcp/cleanup-processes")
        print("Cleanup result:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    print("=== MCP Process Management Test ===")
    test_multiple_requests()
    test_cleanup()
    print("\n=== Test Complete ===")
