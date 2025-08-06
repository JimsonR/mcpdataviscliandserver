import asyncio
import os
from dotenv import load_dotenv
from fastmcp.client.transports import StdioTransport
from fastmcp.client import Client

# For LLM and agent
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Hardcoded Playwright MCP server config (as would be returned by get_server_cfg("playwright"))
server_cfg = {
    "command": "npx",
    "args": [
        "@playwright/mcp@latest",
        "--browser",
        "chrome"
    ],
    "transport": "stdio"
}

async def main():

    # Use StdioTransport for local stdio-based MCP server
    transport = StdioTransport(server_cfg["command"], server_cfg["args"])
    print("[DEBUG] Starting MCP client...")

    async with Client(transport) as client:
        print("[DEBUG] Connected to MCP client. Testing direct tool call before agent...")
        # ...existing code...

        # Try to get tools from MCP server
        try:
            tools = await client.list_tools()
            tools = [t.model_dump() if hasattr(t, 'model_dump') else t for t in tools]
            for tool in tools:
                if 'parameters' not in tool:
                    if 'inputSchema' in tool:
                        tool['parameters'] = tool['inputSchema']
                    else:
                        tool['parameters'] = {"type": "object", "properties": {}}
            print(f"[DEBUG] Tools loaded from MCP server: {[t['name'] for t in tools]}")
            print(f"[DEBUG] Full tool schema: {tools}")
        except Exception as e:
            print(f"[ERROR] Could not list tools: {e}")
            return

        print("[DEBUG] Creating LLM and agent...")
        try:
            llm = AzureChatOpenAI(
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                openai_api_type="azure",
            )
            agent = create_react_agent(llm, tools)
        except Exception as e:
            print(f"[ERROR] Error creating LLM or agent: {e}")
            return
        print("[DEBUG] Agent created. Sending commands...")

        # --- AGENT TOOL EXECUTION LOOP PATCH ---
        async def tool_executor(tool_name, tool_args):
            print(f"[DEBUG] Executing tool: {tool_name} with args: {tool_args}")
            try:
                result = await client.call_tool(tool_name, tool_args)
                print(f"[DEBUG] Tool '{tool_name}' result: {result}")
                return result
            except Exception as e:
                print(f"[ERROR] Tool '{tool_name}' failed: {e}")
                return {"error": str(e)}

        async def run_agent_with_tools(user_message):
            # Start the agent with the user message
            messages = [{"role": "user", "content": user_message}]
            while True:
                response = await agent.ainvoke({"messages": messages})
                print(f"[DEBUG] Agent response: {response}")
                # Find tool calls in the response
                tool_calls = []
                for msg in response.get('messages', []):
                    tc = getattr(msg, 'tool_calls', None)
                    if tc:
                        tool_calls.extend(tc)
                if not tool_calls:
                    # No tool calls, print final message
                    for msg in response.get('messages', []):
                        if hasattr(msg, 'content'):
                            print(f"[AGENT FINAL OUTPUT]: {msg.content}")
                    break
                # Execute each tool and append results as function messages
                for tc in tool_calls:
                    tool_name = tc.get('name')
                    tool_args = tc.get('args', {})
                    result = await tool_executor(tool_name, tool_args)
                    # Add the tool result as a function message for next agent step
                    messages.append({
                        "role": "function",
                        "name": tool_name,
                        "content": str(result)
                    })

        # 1. Open Google
        try:
            await asyncio.wait_for(run_agent_with_tools("open google"), timeout=60)
        except Exception as e:
            print(f"[ERROR] Timeout or error on 'open google': {e}")

        # 2. Take screenshot
        try:
            await asyncio.wait_for(run_agent_with_tools("take screenshot"), timeout=60)
        except Exception as e:
            print(f"[ERROR] Timeout or error on 'take screenshot': {e}")

        # # 3. Close browser to ensure clean exit
        # try:
        #     await asyncio.wait_for(run_agent_with_tools("close browser"), timeout=30)
        # except Exception as e:
        #     print(f"[ERROR] Timeout or error on 'close browser': {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        import sys
        sys.exit(0)  # Ensure clean exit