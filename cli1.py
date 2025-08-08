import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import Client
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import Tool
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import asyncio
import time
import traceback
import tiktoken
import re
from langchain_mcp_adapters.tools import load_mcp_tools

import json
from fastapi.responses import StreamingResponse

from langgraph.prebuilt import create_react_agent
import yaml
from threading import Lock
from sessions.redis_backend import (
    get_chat_history,
    save_chat_history,
    delete_chat_session,
    list_chat_sessions
)


from agent.custom_agent import StructuredAgent
import types
from fastmcp.client.transports import StdioTransport


load_dotenv(".env")

# Path to the YAML config file
MCP_SERVERS_YAML = os.path.join(os.path.dirname(__file__), "mcp_servers.yaml")

# Thread-safe lock for config file access
_mcp_servers_lock = Lock()

def load_mcp_servers():
    with _mcp_servers_lock:
        if not os.path.exists(MCP_SERVERS_YAML):
            return {}
        with open(MCP_SERVERS_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Handle nested 'servers' key structure
        if 'servers' in data:
            return data['servers']
        return data

def save_mcp_servers(servers):
    with _mcp_servers_lock:
        with open(MCP_SERVERS_YAML, "w", encoding="utf-8") as f:
            # Maintain nested 'servers' structure
            yaml.safe_dump({"servers": servers}, f)


# In-memory cache of servers (reload on every change)
_cached_servers = None
_cached_servers_mtime = None

# Cache for server reachability status (expires after 60 seconds)
_server_health_cache = {}
_health_cache_timeout = 60  # seconds

import time

def get_mcp_servers():
    global _cached_servers, _cached_servers_mtime
    try:
        mtime = os.path.getmtime(MCP_SERVERS_YAML)
    except Exception:
        mtime = None
    if _cached_servers is None or mtime != _cached_servers_mtime:
        _cached_servers = load_mcp_servers()
        _cached_servers_mtime = mtime
    return _cached_servers

async def check_server_health(server_name: str, server_config: dict) -> bool:
    """Check if a server is reachable, with caching to avoid repeated checks."""
    current_time = time.time()
    # Use url if present, else use command+args for cache key
    if "url" in server_config:
        cache_key = f"{server_name}_{server_config['url']}"
    elif "command" in server_config:
        cache_key = f"{server_name}_{server_config['command']} {' '.join(server_config.get('args', []))}"
    else:
        cache_key = server_name

    # Check cache first
    if cache_key in _server_health_cache:
        cached_result, timestamp = _server_health_cache[cache_key]
        if current_time - timestamp < _health_cache_timeout:
            return cached_result

    # Perform health check
    try:
        async with get_server_client(server_config) as client:
            await client.list_tools()
            _server_health_cache[cache_key] = (True, current_time)
            return True
    except Exception:
        _server_health_cache[cache_key] = (False, current_time)
        return False

async def get_reachable_servers(servers: dict, skip_health_check: bool = False) -> dict:
    """Get reachable servers, with option to skip health checks for faster responses."""
    if skip_health_check:
        # Return all servers without checking - much faster
        return servers

    reachable_servers = {}
    for server_name, server_config in servers.items():
        if await check_server_health(server_name, server_config):
            reachable_servers[server_name] = server_config
        else:
            # Print a warning, but avoid KeyError if 'url' is missing
            url_or_cmd = server_config.get('url') or server_config.get('command') or 'unknown'
            print(f"Warning: MCP server '{server_name}' at {url_or_cmd} is unreachable")

    return reachable_servers


# For backward compatibility, fallback to hardcoded if YAML missing
def get_server_cfg(server):
    servers = get_mcp_servers()
    if server not in servers:
        raise HTTPException(status_code=404, detail=f"Server '{server}' not found.")
    return servers[server]

def get_server_client(server_cfg):
    if "url" in server_cfg:
        return Client(server_cfg["url"])
    elif "command" in server_cfg and "args" in server_cfg:
        from fastmcp.client.transports import StdioTransport
        # Pass command and args as separate arguments
        transport = StdioTransport(server_cfg["command"], server_cfg["args"])
        return Client(transport)
    else:
        raise HTTPException(status_code=500, detail="Server config must have either 'url' or 'command' and 'args'.")
app = FastAPI()

# Allow CORS for all origins (for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health Check Endpoint ---
@app.get("/api/health")
def health_check():
    """Health check endpoint for backend availability."""
    return {"status": "ok"}


# LangChain Azure OpenAI LLM
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_type="azure",
)



from typing import List, Optional
from uuid import uuid4
from fastapi import Body

# --- In-memory chat history store for tabbed chat (chat_id) ---
chat_histories = {}  # {chat_id: [history_dicts]}

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = None  # Each dict: {"role": "user"/"assistant", "content": "..."}
    chat_id: Optional[str] = None  # For tabbed chat

# --- Chat session management endpoints  internal ---


# @app.post("/chat/create-session")
# def create_chat_session():
#     chat_id = str(uuid4())
#     chat_histories[chat_id] = []
#     return {"chat_id": chat_id}

# @app.get("/chat/list-sessions")
# def list_chat_sessions():
#     return {"chat_ids": list(chat_histories.keys())}

# @app.get("/chat/get-history/{chat_id}")
# def get_chat_history(chat_id: str):
#     return {"chat_id": chat_id, "history": chat_histories.get(chat_id, [])}

# @app.delete("/chat/delete-session/{chat_id}")
# def delete_chat_session(chat_id: str):
#     if chat_id in chat_histories:
#         del chat_histories[chat_id]
#         return {"deleted": True, "chat_id": chat_id}
#     return {"deleted": False, "chat_id": chat_id, "error": "Not found"}


@app.post("/mcp/call-tool")
async def call_mcp_tool(server: str, tool_name: str, arguments: dict):
    server_cfg = get_server_cfg(server)
    async with get_server_client(server_cfg) as client:
        result = await client.call_tool(tool_name, arguments)
        return result

#----------chat session management endpoints using redis backend for persistence
@app.post("/chat/create-session")
def create_chat_session():
    chat_id = str(uuid4())
    save_chat_history(chat_id, [])
    return {"chat_id": chat_id}

@app.get("/chat/list-sessions")
def list_chat_sessions_endpoint():
    return {"chat_ids": list_chat_sessions()}

@app.get("/chat/get-history/{chat_id}")
def get_chat_history_endpoint(chat_id: str):
    return {"chat_id": chat_id, "history": get_chat_history(chat_id)}

@app.delete("/chat/delete-session/{chat_id}")
def delete_chat_session_endpoint(chat_id: str):
    delete_chat_session(chat_id)
    return {"deleted": True, "chat_id": chat_id}

# --- Optimized: Only check reachable servers on demand, not on every frontend load ---

@app.get("/mcp/list-tools")
async def list_mcp_tools(server: str):
    server_cfg = get_server_cfg(server)
    async with get_server_client(server_cfg) as client:
        tools = await client.list_tools()
        # Adapt tool schemas: if 'inputSchema' exists, copy to 'parameters' for agent compatibility
        adapted_tools = []
        for t in tools:
            tool_dict = t.model_dump() if hasattr(t, 'model_dump') else dict(t)
            if 'inputSchema' in tool_dict and 'parameters' not in tool_dict:
                tool_dict['parameters'] = tool_dict['inputSchema']
            adapted_tools.append(tool_dict)
        return adapted_tools



@app.get("/mcp/list-resources")
async def list_mcp_resources(server: str):
    server_cfg = get_server_cfg(server)
    async with get_server_client(server_cfg) as client:
        resources = await client.list_resources()
        return resources

# --- New endpoint: Get resource content by URI ---

@app.get("/mcp/get-resource-content")
async def get_resource_content(server: str, uri: str):
    """Fetch and return the content of a resource by its URI."""
    server_cfg = get_server_cfg(server)
    async with get_server_client(server_cfg) as client:
        content_list = await client.read_resource(uri)
        result = []
        for item in content_list:
            entry = {"mimeType": getattr(item, "mimeType", None)}
            if hasattr(item, "text") and item.text is not None:
                entry["type"] = "text"
                entry["content"] = item.text
            elif hasattr(item, "blob") and item.blob is not None:
                entry["type"] = "binary"
                entry["content"] = f"<binary: {len(item.blob)} bytes>"
            else:
                entry["type"] = "unknown"
                entry["content"] = None
            result.append(entry)
        return result

@app.get("/mcp/list-prompts")
async def list_mcp_prompts(server: str):
    server_cfg = get_server_cfg(server)
    async with get_server_client(server_cfg) as client:
        prompts = await client.list_prompts()
        return prompts
@app.post("/mcp/get-prompt-content")
async def get_prompt_content(server: str, prompt_name: str, arguments: dict = {}):
    server_cfg = get_server_cfg(server)
    async with get_server_client(server_cfg) as client:
        result = await client.get_prompt(prompt_name, arguments)
        # Return all messages as a list of dicts
        return [
            {"role": m.role, "content": getattr(m.content, "text", m.content)}
            for m in result.messages
        ]


from langchain_core.messages import AIMessage



# --- Shared chat history management function ---
# def get_and_update_chat_history(chat_id: str, req_history: list, user_message: str = None, assistant_message: str = None):
#     """
#     Retrieve and update chat history for a given chat_id.
#     - If req_history is provided and longer than stored, use it.
#     - Always append user_message and assistant_message if provided.
#     Returns the updated history.
#     """
#     if chat_id not in chat_histories:
#         chat_histories[chat_id] = []
#     stored_history = chat_histories[chat_id]
#     history = req_history
#     # If req_history is a list of structured entries, use it if longer
#     if history is not None and len(history) > len(stored_history):
#         chat_histories[chat_id] = history
#         stored_history = history
#     else:
#         history = stored_history
#     # If user_message and assistant_message are provided, append as a simple message (legacy)
#     if user_message is not None and assistant_message is not None:
#         updated = (history or []) + [
#             {"role": "user", "content": user_message},
#             {"role": "assistant", "content": assistant_message}
#         ]
#         chat_histories[chat_id] = updated
#         return updated
#     return history or []

#------------Redis-based chat history management function
def get_and_update_chat_history(chat_id: str, req_history: list, user_message: str = None, assistant_message: str = None):
    stored_history = get_chat_history(chat_id)
    history = req_history
    if history is not None and len(history) > len(stored_history):
        save_chat_history(chat_id, history)
        stored_history = history
    else:
        history = stored_history
    if user_message is not None and assistant_message is not None:
        updated = (history or []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        save_chat_history(chat_id, updated)
        return updated
    return history or []

# --- Updated /llm/chat endpoint to support tabbed chat (chat_id) ---
@app.post("/llm/chat")
async def llm_chat(req: ChatRequest):
    # Use shared function for chat history management
    history = req.history
    if req.chat_id:
        history = get_and_update_chat_history(req.chat_id, req.history)
    # Build message list: history + new message
    messages = []
    if history:
        for m in history:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=req.message))
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: llm.invoke(messages))
    # Update chat history if chat_id is used
    if req.chat_id:
        get_and_update_chat_history(req.chat_id, history, req.message, response.content)
    return {"response": response.content, "chat_id": req.chat_id}


# -------------------------------------------------------------------------------------------------------------------------------------#
#                                           Streaming endpoint for LLM/agent responses                                                 #
#--------------------------------------------------------------------------------------------------------------------------------------#
@app.post("/llm/agent-stream")
async def llm_agent_stream(req: ChatRequest):
    """
    Streaming endpoint for agent responses. Streams reasoning/response in real time.
    """
    servers = get_mcp_servers()
    if not servers:
        async def error_stream():
            yield "No MCP servers configured.\n"
        return StreamingResponse(error_stream(), media_type="text/plain")
    try:
        reachable_servers = await get_reachable_servers(servers, skip_health_check=False)
        if not reachable_servers:
            async def error_stream():
                yield "No MCP servers are currently reachable.\n"
            return StreamingResponse(error_stream(), media_type="text/plain")
        # --- Ensure all MCP server connections are initialized before streaming ---
        client = MultiServerMCPClient(reachable_servers)
        # Do not proactively initialize all server connections; let agent/tool invocation handle it (as in non-streaming endpoints)
        tools = await client.get_tools()
        if not tools:
            async def error_stream():
                yield "No tools available from reachable MCP servers.\n"
            return StreamingResponse(error_stream(), media_type="text/plain")
        agent = create_react_agent(llm, tools)
        # --- Use shared chat history logic ---
        history = req.history
        if req.chat_id:
            history = get_and_update_chat_history(req.chat_id, req.history)
        # Limit history to last 10 messages to prevent context overflow
        recent_history = history[-10:] if history and len(history) > 10 else history
        messages = []
        if recent_history:
            for m in recent_history:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    messages.append(AIMessage(content=m["content"]))
        messages.append(HumanMessage(content=req.message))

        import logging
        import traceback
        import sys
        def format_exception_recursive(exc, prefix=""):  # Helper for streaming all sub-exceptions
            import types
            lines = []
            # Print the main exception
            lines.append(f"{prefix}[Error] {type(exc).__name__}: {exc}")
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            lines.extend([prefix + l.rstrip() for l in tb])
            # Python 3.11+ ExceptionGroup/BaseExceptionGroup
            if hasattr(exc, 'exceptions') and isinstance(exc, BaseException):
                for idx, sub in enumerate(getattr(exc, 'exceptions', [])):
                    lines.append(f"{prefix}--- Sub-exception {idx+1} ---")
                    lines.extend(format_exception_recursive(sub, prefix + "    "))
            # Python <3.11: check __cause__ and __context__
            if getattr(exc, '__cause__', None):
                lines.append(f"{prefix}Caused by:")
                lines.extend(format_exception_recursive(exc.__cause__, prefix + "    "))
            if getattr(exc, '__context__', None):
                lines.append(f"{prefix}During handling of the above exception, another exception occurred:")
                lines.extend(format_exception_recursive(exc.__context__, prefix + "    "))
            return lines

        import json

        async def agent_stream():
            chunk_count = 0
            content_count = 0
            try:
                print(f"[DEBUG] Starting agent.astream...")
                # Track the agent's execution state
                current_thought = ""
                current_action = ""
                current_input = ""
                async for chunk in agent.astream({"messages": messages}, config={"recursion_limit": 30, "max_execution_time": 30}):
                    chunk_count += 1
                    print(f"[DEBUG] Chunk {chunk_count}: type={type(chunk)}")
                    print(f"[DEBUG] Chunk {chunk_count}: {chunk}")
                    # Stream all agent reasoning and tool interactions
                    content_lines = []
                    # Handle the nested agent/tools structure
                    if isinstance(chunk, dict):
                        # Check for agent messages (reasoning and responses)
                        if 'agent' in chunk and 'messages' in chunk['agent']:
                            for message in chunk['agent']['messages']:
                                if hasattr(message, 'content') and message.content:
                                    content = str(message.content)
                                    print(f"[DEBUG] Found agent message content: {content}")
                                    # Check if this looks like structured agent reasoning
                                    if content.startswith('Thought:') or content.startswith('Action:') or content.startswith('Action Input:'):
                                        content_lines.append(content)
                                    else:
                                        # Regular agent response - add as thinking if it's reasoning
                                        if len(content) > 20 and not content.startswith('Based on'):
                                            content_lines.append(f"Thought: {content}")
                                        else:
                                            content_lines.append(content)

                                # ✅ NEW: Check for tool calls in the agent message
                                if hasattr(message, 'tool_calls') and message.tool_calls:
                                    for tool_call in message.tool_calls:
                                        tool_name = tool_call.get('name', 'unknown')
                                        tool_args = tool_call.get('args', {})
                                        content_lines.append(f"Action: {tool_name}")
                                        if tool_args:
                                            content_lines.append(f"Action Input: {json.dumps(tool_args)}")
                                        print(f"[DEBUG] Found tool call in agent message: {tool_name}")

                                # ✅ NEW: Check for tool calls in additional_kwargs (OpenAI format)
                                if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
                                    tool_calls = message.additional_kwargs.get('tool_calls', [])
                                    for tool_call in tool_calls:
                                        if tool_call.get('type') == 'function':
                                            func = tool_call.get('function', {})
                                            tool_name = func.get('name', 'unknown')
                                            tool_args = func.get('arguments', '{}')
                                            content_lines.append(f"Action: {tool_name}")
                                            if tool_args and tool_args != '{}':
                                                content_lines.append(f"Action Input: {tool_args}")
                                            print(f"[DEBUG] Found tool call in additional_kwargs: {tool_name}")
                        # Check for tool messages (tool calls and outputs) - CAPTURE THESE
                        elif 'tools' in chunk and 'messages' in chunk['tools']:
                            for message in chunk['tools']['messages']:
                                if hasattr(message, 'content') and message.content:
                                    tool_content = str(message.content)
                                    print(f"[DEBUG] Found tool message: {tool_content[:100]}...")
                                    # Format as observation for frontend
                                    content_lines.append(f"Observation: {tool_content}")
                        # Handle tool calls directly from the chunk (fallback)
                        elif 'action' in chunk or 'tool_calls' in chunk:
                            if 'action' in chunk:
                                action = chunk['action']
                                if hasattr(action, 'tool'):
                                    tool_name = getattr(action, 'tool', 'unknown')
                                    tool_input = getattr(action, 'tool_input', {})
                                    content_lines.append(f"Action: {tool_name}")
                                    if tool_input:
                                        content_lines.append(f"Action Input: {json.dumps(tool_input)}")
                                    print(f"[DEBUG] Found action: {tool_name}")
                            # Handle LangChain tool calls format
                            if 'tool_calls' in chunk:
                                for tool_call in chunk['tool_calls']:
                                    if hasattr(tool_call, 'name'):
                                        tool_name = tool_call.name
                                        tool_args = getattr(tool_call, 'args', {})
                                        content_lines.append(f"Action: {tool_name}")
                                        if tool_args:
                                            content_lines.append(f"Action Input: {json.dumps(tool_args)}")
                                        print(f"[DEBUG] Found tool call: {tool_name}")
                        # Handle agent thoughts/outcomes (fallback)
                        elif 'agent_outcome' in chunk:
                            outcome = chunk['agent_outcome']
                            if hasattr(outcome, 'log') and outcome.log:
                                content_lines.append(f"Thought: {outcome.log}")
                                print(f"[DEBUG] Found agent thought: {outcome.log}")
                        # Handle intermediate steps (fallback)
                        elif 'intermediate_steps' in chunk:
                            steps = chunk['intermediate_steps']
                            for step in steps:
                                if hasattr(step, 'action') and hasattr(step, 'observation'):
                                    action = step.action
                                    if hasattr(action, 'tool'):
                                        tool_name = action.tool
                                        tool_input = getattr(action, 'tool_input', {})
                                        content_lines.append(f"Action: {tool_name}")
                                        if tool_input:
                                            content_lines.append(f"Action Input: {json.dumps(tool_input)}")
                                    observation = step.observation
                                    if observation:
                                        content_lines.append(f"Observation: {observation}")
                        # Fallback to direct content extraction
                        elif 'content' in chunk:
                            content_lines.append(str(chunk['content']))
                            print(f"[DEBUG] Found direct content: {chunk['content']}")
                    elif hasattr(chunk, 'content'):
                        content_lines.append(str(chunk.content))
                        print(f"[DEBUG] Found direct chunk content: {chunk.content}")
                    # Also check if the chunk has tool-related attributes directly
                    elif hasattr(chunk, 'tool_calls'):
                        for tool_call in chunk.tool_calls:
                            if hasattr(tool_call, 'function'):
                                func = tool_call.function
                                content_lines.append(f"Action: {func.name}")
                                if hasattr(func, 'arguments'):
                                    content_lines.append(f"Action Input: {func.arguments}")
                    # DEBUG: Print what we're about to stream
                    if content_lines:
                        print(f"[DEBUG] About to stream {len(content_lines)} content lines:")
                        for i, line in enumerate(content_lines):
                            print(f"[DEBUG]   Line {i}: {line[:100]}...")
                    else:
                        print(f"[DEBUG] No content lines found in this chunk")
                    # Stream each content line separately for real-time updates
                    for content in content_lines:
                        if content and content.strip():
                            content_count += 1
                            json_chunk = {
                                "type": "content",
                                "data": content.strip()
                            }
                            chunk_json = json.dumps(json_chunk) + "\n"
                            print(f"[DEBUG] Yielding content {content_count}: {chunk_json.strip()}")
                            yield chunk_json
                    # Small delay between chunks to prevent overwhelming the frontend
                    await asyncio.sleep(0.01)
                print(f"[DEBUG] Stream finished. Processed {chunk_count} chunks, yielded {content_count} content chunks")
                yield json.dumps({"type": "end"}) + "\n"
            except Exception as e_inner:
                print(f"[DEBUG] Exception: {e_inner}")
                import traceback
                traceback.print_exc()
                error_chunk = {
                    "type": "error",
                    "data": f"Error: {str(e_inner)}"
                }
                yield json.dumps(error_chunk) + "\n"

        return StreamingResponse(agent_stream(), media_type="application/json")
    except Exception as e:
        import traceback
        def format_exception_recursive(exc, prefix=""):
            import types
            lines = []
            lines.append(f"{prefix}[Error] {type(exc).__name__}: {exc}")
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            lines.extend([prefix + l.rstrip() for l in tb])
            if hasattr(exc, 'exceptions') and isinstance(exc, BaseException):
                for idx, sub in enumerate(getattr(exc, 'exceptions', [])):
                    lines.append(f"{prefix}--- Sub-exception {idx+1} ---")
                    lines.extend(format_exception_recursive(sub, prefix + "    "))
            if getattr(exc, '__cause__', None):
                lines.append(f"{prefix}Caused by:")
                lines.extend(format_exception_recursive(exc.__cause__, prefix + "    "))
            if getattr(exc, '__context__', None):
                lines.append(f"{prefix}During handling of the above exception, another exception occurred:")
                lines.extend(format_exception_recursive(exc.__context__, prefix + "    "))
            return lines
        async def error_stream(exc):
            for line in format_exception_recursive(exc):
                yield line + "\n"
        return StreamingResponse(error_stream(e), media_type="text/plain")
#--------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------

@app.post("/llm/agent")
async def llm_agent(req: ChatRequest):
    servers = get_mcp_servers()
    # Safety check: ensure we have servers configured
    if not servers:
        return {"response": "No MCP servers configured.", "error": True}
    try:
        # Skip health checks for faster response
        reachable_servers = await get_reachable_servers(servers, skip_health_check=True)
        if not reachable_servers:
            return {"response": "No MCP servers configured.", "error": True}
        print(f"Using {len(reachable_servers)} servers: {list(reachable_servers.keys())}")
        client = MultiServerMCPClient(reachable_servers)
        tools = await client.get_tools()
        # Safety check: ensure we have tools
        if not tools:
            # If no tools, try fallback with health check
            reachable_servers = await get_reachable_servers(servers, skip_health_check=False)
            if not reachable_servers:
                return {"response": "No MCP servers are currently reachable. Please check if your MCP servers are running.", "error": True}
            client = MultiServerMCPClient(reachable_servers)
            tools = await client.get_tools()
            if not tools:
                return {"response": "No tools available from reachable MCP servers.", "error": True}
        agent = create_react_agent(llm, tools)
        # --- Use shared chat history logic ---
        history = req.history
        if req.chat_id:
            history = get_and_update_chat_history(req.chat_id, req.history)
        # Limit history to last 10 messages to prevent context overflow
        recent_history = history[-10:] if history and len(history) > 10 else history
        messages = []
        if recent_history:
            for m in recent_history:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    messages.append(AIMessage(content=m["content"]))
        messages.append(HumanMessage(content=req.message))
        # Add recursion limit and timeout
        result = await agent.ainvoke(
            {"messages": messages},
            config={
                "recursion_limit": 5,
                "max_execution_time": 30
            }
        )
        # Update chat history if chat_id is used
        if req.chat_id:
            get_and_update_chat_history(req.chat_id, history, req.message, result['messages'][-1].content)
        return {"response": result['messages'][-1].content}
    except Exception as e:
        error_msg = str(e)
        print(f"Agent error: {error_msg}")
        if "recursion" in error_msg.lower():
            return {"response": "Agent hit recursion limit. The task may be too complex or tools are failing repeatedly.", "error": True}
        return {"response": f"Agent error: {error_msg}", "error": True}

    

@app.post("/llm/agent-detailed")
async def llm_agent_detailed(req: ChatRequest):
    servers = get_mcp_servers()
    # Safety check: ensure we have servers configured
    if not servers:
        return {
            "response": "No MCP servers configured.",
            "error": True,
            "tool_executions": [],
            "full_conversation": []
        }
    try:
        # Skip health checks for faster response
        reachable_servers = await get_reachable_servers(servers, skip_health_check=True)
        if not reachable_servers:
            return {
                "response": "No MCP servers configured.",
                "error": True,
                "tool_executions": [],
                "full_conversation": []
            }
        print(f"Using {len(reachable_servers)} servers: {list(reachable_servers.keys())}")
        client = MultiServerMCPClient(reachable_servers)
        # Try to get tools with fallback
        try:
            tools = await client.get_tools()
        except Exception:
            # Fallback with health check
            reachable_servers = await get_reachable_servers(servers, skip_health_check=False)
            if not reachable_servers:
                return {
                    "response": "No MCP servers are currently reachable. Please check if your MCP servers are running.",
                    "error": True,
                    "tool_executions": [],
                    "full_conversation": []
                }
            client = MultiServerMCPClient(reachable_servers)
            tools = await client.get_tools()
        # Safety check: ensure we have tools
        if not tools:
            return {
                "response": "No tools available from reachable MCP servers.",
                "error": True,
                "tool_executions": [],
                "full_conversation": []
            }
        agent = create_react_agent(llm, tools)
        # --- Use shared chat history logic ---
        history = req.history
        if req.chat_id:
            history = get_and_update_chat_history(req.chat_id, req.history)
        # Limit history to last 10 messages to prevent context overflow
        recent_history = history[-10:] if history and len(history) > 10 else history
        messages = []
        if recent_history:
            for m in recent_history:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    messages.append(AIMessage(content=m["content"]))
        messages.append(HumanMessage(content=req.message))
        # Add recursion limit and timeout
        result = await agent.ainvoke(
            {"messages": messages},
            config={
                "recursion_limit": 30,
                "max_execution_time": 60
            }
        )
        # Update chat history if chat_id is used
        if req.chat_id:
            get_and_update_chat_history(req.chat_id, history, req.message, result['messages'][-1].content)
        # Extract tool execution details from the conversation
        tool_executions = []
        for message in result['messages']:
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_executions.append({
                        "tool_name": tool_call.get("name", "unknown"),
                        "arguments": tool_call.get("args", {}),
                        "id": tool_call.get("id", "unknown")
                    })
            elif hasattr(message, 'type') and message.type == "tool":
                tool_executions.append({
                    "tool_response": getattr(message, 'content', 'No content'),
                    "tool_call_id": getattr(message, 'tool_call_id', 'unknown')
                })
        return {
            "response": result['messages'][-1].content,
            "tool_executions": tool_executions,
            "full_conversation": [
                {
                    "type": getattr(m, 'type', 'unknown'),
                    "content": getattr(m, 'content', str(m)),
                    "role": getattr(m, 'role', 'unknown') if hasattr(m, 'role') else None
                }
                for m in result['messages']
            ]
        }
    except Exception as e:
        error_msg = str(e)
        print(f"Agent detailed error: {error_msg}")
        if "recursion" in error_msg.lower():
            return {
                "response": "Agent hit recursion limit. The task may be too complex or tools are failing repeatedly.",
                "error": True,
                "tool_executions": [],
                "full_conversation": []
            }
        return {
            "response": f"Agent error: {error_msg}",
            "error": True,
            "tool_executions": [],
            "full_conversation": []
        }




# --- New endpoint: Enhanced StructuredAgent with Claude-like response formatting ---
@app.post("/llm/structured-agent")
async def llm_structured_agent(req: ChatRequest):
    """
    Chat endpoint using the enhanced StructuredAgent with structured response formatting.
    This endpoint provides better error handling, contextual tool execution display,
    and Claude-like response formatting.
    """
    servers = get_mcp_servers()
    # Safety check: ensure we have servers configured
    if not servers:
        return {
            "response": "No MCP servers configured.",
            "error": True,
            "tool_executions": [],
            "reasoning_steps": [],
            "formatted_output": "No MCP servers configured.",
            "agent_type": "structured"
        }
    try:
        # Skip health checks for faster initial response - servers will fail gracefully if unreachable
        reachable_servers = await get_reachable_servers(servers, skip_health_check=True)
        if not reachable_servers:
            return {
                "response": "No MCP servers configured.",
                "error": True,
                "tool_executions": [],
                "reasoning_steps": [],
                "formatted_output": "No MCP servers configured.",
                "agent_type": "structured"
            }
        print(f"Using {len(reachable_servers)} servers: {list(reachable_servers.keys())}")
        client = MultiServerMCPClient(reachable_servers)
        # Try to get tools - if any servers are unreachable, this will filter them out
        try:
            tools = await client.get_tools()
        except Exception as e:
            print(f"Error getting tools, falling back to health check: {e}")
            # Fallback: do health checks if getting tools fails
            reachable_servers = await get_reachable_servers(servers, skip_health_check=False)
            if not reachable_servers:
                return {
                    "response": "No MCP servers are currently reachable. Please check if your MCP servers are running.",
                    "error": True,
                    "tool_executions": [],
                    "reasoning_steps": [],
                    "formatted_output": "No MCP servers are currently reachable. Please check if your MCP servers are running.",
                    "agent_type": "structured"
                }
            client = MultiServerMCPClient(reachable_servers)
            tools = await client.get_tools()
        if not tools:
            return {
                "response": "No tools available from reachable MCP servers.",
                "error": True,
                "tool_executions": [],
                "reasoning_steps": [],
                "formatted_output": "No tools available from reachable MCP servers.",
                "agent_type": "structured"
            }
        # Initialize the enhanced agent
        agent = StructuredAgent(llm, tools)
        # --- Use shared chat history logic ---
        history = req.history
        if req.chat_id:
            history = get_and_update_chat_history(req.chat_id, req.history)
        # Limit history to last 10 messages to prevent context overflow
        recent_history = history[-10:] if history and len(history) > 10 else history
        messages = []
        if recent_history:
            for m in recent_history:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    from langchain_core.messages import AIMessage
                    messages.append(AIMessage(content=m["content"]))
        messages.append(HumanMessage(content=req.message))
        # Use the enhanced invoke method
        result = await agent.invoke(messages)
        # Extract reasoning steps for detailed breakdown
        reasoning_steps = result.get("reasoning_steps", [])
        # Format tool executions in the expected format (for backward compatibility)
        tool_executions = []
        for step in reasoning_steps:
            for tool_result in step.get("tool_results", []):
                tool_executions.append({
                    "tool_name": tool_result["tool_name"],
                    "arguments": tool_result["arguments"],
                    "result": tool_result["result"][:500] + "..." if len(str(tool_result["result"])) > 500 else tool_result["result"]
                })
        # Build the full structured entry for this turn
        structured_entry = {
            "user_message": req.message,
            "response": result.get("response"),
            "formatted_output": result.get("formatted_output"),
            "reasoning_steps": reasoning_steps,
            "tool_executions": tool_executions,
            "iterations": result.get("iterations"),
            "messages": [getattr(m, "content", str(m)) for m in result.get("messages", [])] if "messages" in result else [],
            "success": True,
            "error": False,
            "agent_type": "structured"
        }
        # Update chat history if chat_id is used (append structured entry)
        # if req.chat_id:
        #     if req.chat_id not in chat_histories:
        #         chat_histories[req.chat_id] = []
        #     chat_histories[req.chat_id].append(structured_entry)
        if req.chat_id:
            history = get_chat_history(req.chat_id)
            history.append(structured_entry)
            save_chat_history(req.chat_id, history)
        return structured_entry
    except Exception as e:
        error_msg = str(e)
        print(f"Structured agent error: {error_msg}")
        return {
            "response": f"Structured agent error: {error_msg}",
            "formatted_output": f"Structured agent error: {error_msg}",
            "error": True,
            "error_type": type(e).__name__,
            "tool_executions": [],
            "reasoning_steps": [],
            "iterations": 0,
            "agent_type": "structured"
        }


# Optional: Add a new endpoint specifically for the formatted output
@app.post("/llm/structured-agent-formatted")
async def llm_structured_agent_formatted(req: ChatRequest):
    """
    Alternative endpoint that returns just the formatted output for direct display.
    This is useful if you want to display the response exactly like Claude does.
    """
    result = await llm_structured_agent(req)
    
    if result.get("error"):
        return {"formatted_response": result["response"], "error": True}
    
    return {
        "formatted_response": result.get("formatted_output", result["response"]),
        "error": False,
        "iterations": result.get("iterations", 0),
        "tool_count": len(result.get("tool_executions", []))
    }



# --- Groq LLM integration for testing ---
from langchain_groq import ChatGroq

@app.post("/llm/groq-structured-agent")
async def groq_structured_agent(req: ChatRequest):
    """
    Chat endpoint using StructuredAgent with Groq LLM for testing.
    """
    servers = get_mcp_servers()
    if not servers:
        return {
            "response": "No MCP servers configured.",
            "error": True,
            "tool_executions": [],
            "reasoning_steps": [],
            "formatted_output": "No MCP servers configured.",
            "agent_type": "structured-groq"
        }
    try:
        reachable_servers = await get_reachable_servers(servers, skip_health_check=True)
        if not reachable_servers:
            return {
                "response": "No MCP servers configured.",
                "error": True,
                "tool_executions": [],
                "reasoning_steps": [],
                "formatted_output": "No MCP servers configured.",
                "agent_type": "structured-groq"
            }
        print(f"Using {len(reachable_servers)} servers: {list(reachable_servers.keys())}")
        client = MultiServerMCPClient(reachable_servers)
        try:
            tools = await client.get_tools()
        except Exception as e:
            print(f"Error getting tools, falling back to health check: {e}")
            reachable_servers = await get_reachable_servers(servers, skip_health_check=False)
            if not reachable_servers:
                return {
                    "response": "No MCP servers are currently reachable. Please check if your MCP servers are running.",
                    "error": True,
                    "tool_executions": [],
                    "reasoning_steps": [],
                    "formatted_output": "No MCP servers are currently reachable. Please check if your MCP servers are running.",
                    "agent_type": "structured-groq"
                }
            client = MultiServerMCPClient(reachable_servers)
            tools = await client.get_tools()
        if not tools:
            return {
                "response": "No tools available from reachable MCP servers.",
                "error": True,
                "tool_executions": [],
                "reasoning_steps": [],
                "formatted_output": "No tools available from reachable MCP servers.",
                "agent_type": "structured-groq"
            }
        # --- Groq LLM setup ---
        groq_llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "qwen-qwq-32b"),
        )
        agent = create_react_agent(groq_llm, tools)

        result = await agent.ainvoke({"messages": [{"role": "user", "content": req.message}]})
        return result
    #     messages = []
    #     if req.history:
    #         recent_history = req.history[-10:] if len(req.history) > 10 else req.history
    #         for m in recent_history:
    #             if m["role"] == "user":
    #                 messages.append(HumanMessage(content=m["content"]))
    #             elif m["role"] == "assistant":
    #                 messages.append(AIMessage(content=m["content"]))
    #     messages.append(HumanMessage(content=req.message))
    #     result = await agent.invoke(messages)
    #     reasoning_steps = result.get("reasoning_steps", [])
    #     tool_executions = []
    #     for step in reasoning_steps:
    #         for tool_result in step.get("tool_results", []):
    #             tool_executions.append({
    #                 "tool_name": tool_result["tool_name"],
    #                 "arguments": tool_result["arguments"],
    #                 "result": tool_result["result"][:500] + "..." if len(str(tool_result["result"])) > 500 else tool_result["result"]
    #             })
    #     return {
    #         "response": result.get("response"),
    #         "formatted_output": result.get("formatted_output"),
    #         "reasoning_steps": reasoning_steps,
    #         "tool_executions": tool_executions,
    #         "iterations": result.get("iterations"),
    #         "messages": [getattr(m, "content", str(m)) for m in result.get("messages", [])] if "messages" in result else [],
    #         "success": True,
    #         "error": False,
    #         "agent_type": "structured-groq"
    #     }
    except Exception as e:
        error_msg = str(e)
        print(f"Groq structured agent error: {error_msg}")
        return {
            "response": f"Groq structured agent error: {error_msg}",
            "formatted_output": f"Groq structured agent error: {error_msg}",
            "error": True,
            "error_type": type(e).__name__,
            "tool_executions": [],
            "reasoning_steps": [],
            "iterations": 0,
            "agent_type": "structured-groq"
        }
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
from langchain_openai import ChatOpenAI  # or the correct import for your Mistral client

@app.post("/llm/mistral-structured-agent")
async def mistral_structured_agent(req: ChatRequest):
    """
    Chat endpoint using StructuredAgent with local Mistral 7B Instruct for testing.
    """
    servers = get_mcp_servers()
    if not servers:
        return {
            "response": "No MCP servers configured.",
            "error": True,
            "tool_executions": [],
            "reasoning_steps": [],
            "formatted_output": "No MCP servers configured.",
            "agent_type": "structured-mistral"
        }
    try:
        reachable_servers = await get_reachable_servers(servers, skip_health_check=True)
        if not reachable_servers:
            return {
                "response": "No MCP servers configured.",
                "error": True,
                "tool_executions": [],
                "reasoning_steps": [],
                "formatted_output": "No MCP servers configured.",
                "agent_type": "structured-mistral"
            }
        print(f"Using {len(reachable_servers)} servers: {list(reachable_servers.keys())}")
        client = MultiServerMCPClient(reachable_servers)
        try:
            tools = await client.get_tools()
        except Exception as e:
            print(f"Error getting tools, falling back to health check: {e}")
            reachable_servers = await get_reachable_servers(servers, skip_health_check=False)
            if not reachable_servers:
                return {
                    "response": "No MCP servers are currently reachable.",
                    "error": True,
                    "tool_executions": [],
                    "reasoning_steps": [],
                    "formatted_output": "No MCP servers are currently reachable.",
                    "agent_type": "structured-mistral"
                }
            client = MultiServerMCPClient(reachable_servers)
            tools = await client.get_tools()
        if not tools:
            return {
                "response": "No tools available from reachable MCP servers.",
                "error": True,
                "tool_executions": [],
                "reasoning_steps": [],
                "formatted_output": "No tools available from reachable MCP servers.",
                "agent_type": "structured-mistral"
            }
        # --- Mistral LLM setup ---
        mistral_llm = ChatOpenAI(
            openai_api_key="EMPTY",  # or your key if needed
            openai_api_base="http://localhost:11434/v1",  # adjust to your Mistral endpoint
            model="mistral:7b",  # adjust as needed
        )
        agent = create_react_agent(mistral_llm, tools)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": req.message}]})
        return result
    except Exception as e:
        error_msg = str(e)
        print(f"Mistral structured agent error: {error_msg}")
        return {
            "response": f"Mistral structured agent error: {error_msg}",
            "formatted_output": f"Mistral structured agent error: {error_msg}",
            "error": True,
            "error_type": type(e).__name__,
            "tool_executions": [],
            "reasoning_steps": [],
            "iterations": 0,
            "agent_type": "structured-mistral"
        }
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------


# --- Optimized: Only check reachable servers when explicitly requested ---
@app.get("/langchain/list-tools")
async def langchain_list_tools(servers: str = None, only_reachable: bool = False):
    all_servers = get_mcp_servers()
    if servers:
        selected = {k: v for k, v in all_servers.items() if k in servers.split(",")}
    else:
        selected = all_servers
    if only_reachable:
        # Filter out unreachable servers
        reachable_servers = {}
        for server_name, server_config in selected.items():
            try:
                async with Client(server_config["url"]) as client:
                    await client.list_tools()  # Test if server is reachable
                    reachable_servers[server_name] = server_config
            except Exception as e:
                print(f"Warning: MCP server '{server_name}' at {server_config['url']} is unreachable: {e}")
                continue
        if not reachable_servers:
            return {"error": "No reachable MCP servers found"}
        client = MultiServerMCPClient(reachable_servers)
    else:
        # Do not check reachability, just return all
        client = MultiServerMCPClient(selected)
    tools = await client.get_tools()
    # Convert each tool to a dict (if needed)
    serializable_tools = []
    for tool in tools:
        if hasattr(tool, "dict"):
            serializable_tools.append(tool.dict())
        elif hasattr(tool, "model_dump"):
            serializable_tools.append(tool.model_dump())
        else:
            serializable_tools.append(tool)
    return serializable_tools

# --- MCP Server Config Management Endpoints ---
class MCPServerConfig(BaseModel):
    url: str
    transport: str

@app.get("/mcp/servers")
def list_mcp_servers():
    """List all MCP server names and configs."""
    return get_mcp_servers()

@app.post("/mcp/servers")
def add_mcp_server(name: str, config: MCPServerConfig):
    """Add a new MCP server."""
    servers = get_mcp_servers()
    if name in servers:
        raise HTTPException(status_code=400, detail=f"Server '{name}' already exists.")
    servers[name] = config.dict()
    save_mcp_servers(servers)
    return {"message": f"Server '{name}' added.", "servers": servers}

@app.put("/mcp/servers/{name}")
def update_mcp_server(name: str, config: MCPServerConfig):
    """Update an existing MCP server."""
    servers = get_mcp_servers()
    if name not in servers:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found.")
    servers[name] = config.dict()
    save_mcp_servers(servers)
    return {"message": f"Server '{name}' updated.", "servers": servers}

@app.delete("/mcp/servers/{name}")
def delete_mcp_server(name: str):
    """Delete an MCP server."""
    servers = get_mcp_servers()
    if name not in servers:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found.")
    del servers[name]
    save_mcp_servers(servers)
    return {"message": f"Server '{name}' deleted.", "servers": servers}



# --- New endpoint: Streaming StructuredAgent with formatted output ---
@app.post("/llm/structured-agent-stream")
async def llm_structured_agent_stream(req: ChatRequest):
    """
    Streaming endpoint for StructuredAgent responses. Streams reasoning, tool actions, and tool outputs 
    in a clean, formatted manner similar to MCP reasoning responses.
    """
    servers = get_mcp_servers()
    if not servers:
        async def error_stream():
            yield json.dumps({"type": "error", "data": "No MCP servers configured."}) + "\n"
        return StreamingResponse(error_stream(), media_type="application/json")
    
    try:
        reachable_servers = await get_reachable_servers(servers, skip_health_check=True)
        if not reachable_servers:
            async def error_stream():
                yield json.dumps({"type": "error", "data": "No MCP servers configured."}) + "\n"
            return StreamingResponse(error_stream(), media_type="application/json")
        
        client = MultiServerMCPClient(reachable_servers)
        
        # Use the working approach but add session context for browser tools
        try:
            tools = await client.get_tools()
            print(f"Loaded {len(tools)} tools from all servers")
            if tools:
                tool_names = [tool.name for tool in tools if hasattr(tool, 'name')]
                print(f"Available tools: {tool_names[:10]}...")  # Show first 10 tools
                    
        except Exception as e:
            reachable_servers = await get_reachable_servers(servers, skip_health_check=False)
            if not reachable_servers:
                async def error_stream():
                    yield json.dumps({"type": "error", "data": "No MCP servers are currently reachable. Please check if your MCP servers are running."}) + "\n"
                return StreamingResponse(error_stream(), media_type="application/json")
            client = MultiServerMCPClient(reachable_servers)
            tools = await client.get_tools()
        if not tools:
            async def error_stream():
                yield json.dumps({"type": "error", "data": "No tools available from reachable MCP servers."}) + "\n"
            return StreamingResponse(error_stream(), media_type="application/json")
        agent = StructuredAgent(llm, tools)
        
        # --- Prepare messages from history ---
        history = req.history
        if req.chat_id:
            history = get_and_update_chat_history(req.chat_id, req.history)
        recent_history = history[-10:] if history and len(history) > 10 else history
        messages = []
        if recent_history:
            for m in recent_history:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    from langchain_core.messages import AIMessage
                    messages.append(AIMessage(content=m["content"]))
        messages.append(HumanMessage(content=req.message))

        # --- Structured JSON streaming protocol ---
        async def structured_agent_stream():
            try:
                step_count = 0
                last_thought_content = ""  # Track last thinking content for deduplication
                accumulated_response = ""  # Track the complete assistant response for saving
                
                def clean_content_for_similarity(content):
                    """Clean content for similarity comparison"""
                    import re
                    # Remove markdown headers, extra spaces, and normalize
                    cleaned = re.sub(r'#+\s*', '', content)  # Remove markdown headers
                    cleaned = re.sub(r'\s+', ' ', cleaned)   # Normalize whitespace
                    return cleaned.lower().strip()
                
                def is_content_similar(content1, content2, threshold=0.7):
                    """Check if two content pieces are similar using multiple methods"""
                    if not content1 or not content2:
                        return False
                    
                    # Normalize content for comparison
                    norm1 = clean_content_for_similarity(content1)
                    norm2 = clean_content_for_similarity(content2)
                    
                    # Method 1: Jaccard similarity
                    words1 = set(norm1.split())
                    words2 = set(norm2.split())
                    
                    if words1 and words2:
                        common_words = words1.intersection(words2)
                        all_words = words1.union(words2)
                        jaccard_similarity = len(common_words) / len(all_words)
                        
                        # Method 2: Substring check
                        substring_similarity = 1.0 if (norm1 in norm2 or norm2 in norm1) else 0
                        
                        # Method 3: Length ratio
                        length_ratio = min(len(norm1), len(norm2)) / max(len(norm1), len(norm2))
                        
                        # Return True if content is similar by any method
                        return (jaccard_similarity > threshold or 
                                substring_similarity > 0.8 or 
                                (jaccard_similarity > 0.5 and length_ratio > 0.8))
                    
                    return False

                async for step in agent.astream(messages):
                    if step["type"] == "thought":
                        step_count += 1
                        thought_content = step['content'].strip()
                        
                        if thought_content:
                            last_thought_content = thought_content
                            
                            # Add to accumulated response in the same format as your example
                            thought_section = f"\n## 🤔 Thinking (Step {step_count})\n\n{thought_content}\n\n"
                            accumulated_response += thought_section
                            
                            # Emit structured thought block - frontend will render as markdown
                            yield json.dumps({
                                "type": "thought",
                                "content": thought_content,
                                "step": step_count
                            }) + "\n"
                    
                    elif step["type"] == "tool_execution":
                        tool_name = step.get('tool_name', 'Unknown Tool')
                        
                        def format_tool_result(result):
                            """Format tool result for better display"""
                            if result is None:
                                return None
                            
                            result_str = str(result)
                            # Truncate very long results but preserve structure
                            # if len(result_str) > 2000:
                            #     return result_str[:2000] + "... (truncated)"
                            return result_str
                        
                        def format_arguments(args):
                            """Format arguments for better display"""
                            if not args:
                                return {}
                            
                            # If args is a string, try to parse as JSON for better formatting
                            if isinstance(args, str):
                                try:
                                    import json as json_lib
                                    return json_lib.loads(args)
                                except:
                                    return {"raw": args}
                            return args
                        
                        # Check if this is a multi-tool call
                        if tool_name == "multi_tool_use.parallel" and step.get("arguments"):
                            # Handle parallel multi-tool calls
                            args = step.get("arguments", {})
                            tool_uses = args.get("tool_uses", [])
                            
                            for i, sub_tool in enumerate(tool_uses):
                                recipient_name = sub_tool.get("recipient_name", "unknown")
                                clean_name = recipient_name.replace("functions.", "") if recipient_name.startswith("functions.") else recipient_name
                                sub_params = sub_tool.get("parameters", {})
                                
                                # Emit structured tool use block with enhanced formatting
                                yield json.dumps({
                                    "type": "tool_use",
                                    "tool_name": clean_name,
                                    "arguments": format_arguments(sub_params),
                                    "result": None,  # Will be filled when result comes
                                    "parallel_index": i,
                                    "is_parallel": True,
                                    "status": "executing",
                                    "timestamp": time.time()
                                }) + "\n"
                        else:
                            # Handle single tool calls
                            tool_result = step.get("result")
                            execution_time = step.get("execution_time")
                            error_details = step.get("error")
                            
                            # Format tool execution for accumulated response (matching your example format)
                            tool_args = format_arguments(step.get("arguments", {}))
                            formatted_result = format_tool_result(tool_result)
                            
                            # Add to accumulated response in the same format as your example
                            tool_section = f"\n<details>\n<summary>🔧 <strong>Tool: {tool_name}</strong> ✅</summary>\n\n"
                            
                            if tool_args:
                                import json as json_lib
                                tool_section += f"**Arguments:**\n```json\n{json_lib.dumps(tool_args, indent=2)}\n```\n\n"
                            
                            tool_section += f"**Result:**\n```\n{formatted_result}\n```\n\n</details>\n\n"
                            accumulated_response += tool_section
                            
                            # Enhanced error logging for debugging
                            if error_details or (tool_result and "error" in str(tool_result).lower()):
                                print(f"Tool execution error for {tool_name}:")
                                print(f"  Arguments: {step.get('arguments', {})}")
                                print(f"  Result: {tool_result}")
                                print(f"  Error details: {error_details}")
                                print(f"  Full step: {step}")
                            
                            # Emit structured tool use block with enhanced formatting
                            yield json.dumps({
                                "type": "tool_use",
                                "tool_name": tool_name,
                                "arguments": format_arguments(step.get("arguments", {})),
                                "result": format_tool_result(tool_result),
                                "error_details": error_details,
                                "is_parallel": False,
                                "status": "completed" if tool_result is not None else "executing",
                                "execution_time": execution_time,
                                "timestamp": time.time(),
                                "result_type": "success" if tool_result is not None and "error" not in str(tool_result).lower() else "error"
                            }) + "\n"
                    
                    elif step["type"] == "final_answer":
                        if step.get("content"):
                            final_content = step['content'].strip()
                            
                            # Accumulate the response content for saving
                            if not is_content_similar(last_thought_content, final_content):
                                # Only add if it's not similar to the last thought
                                accumulated_response += final_content
                            
                            # Check for similarity with last thought
                            should_output_final = not is_content_similar(last_thought_content, final_content)
                            
                            if should_output_final:
                                # Emit structured final answer - frontend will render as markdown
                                yield json.dumps({
                                    "type": "final_answer",
                                    "content": final_content
                                }) + "\n"
                    
                    # Small delay between chunks for proper streaming
                    await asyncio.sleep(0.01)
                
                # Emit completion signal
                yield json.dumps({"type": "stream_end"}) + "\n"
                
                # Save chat history after streaming has ended if chat_id is provided
                if req.chat_id:
                    try:
                        # Get current history and append the new interaction
                        current_history = get_chat_history(req.chat_id)
                        
                        # Add user message
                        current_history.append({
                            "role": "user",
                            "content": req.message
                        })
                        
                        # Add assistant response (collect all final_answer content)
                        assistant_response = accumulated_response if accumulated_response else "Response completed via streaming"
                        
                        current_history.append({
                            "role": "assistant", 
                            "content": assistant_response
                        })
                        
                        save_chat_history(req.chat_id, current_history)
                    except Exception as save_error:
                        print(f"Error saving chat history: {save_error}")
                
            except Exception as e:
                # Emit structured error
                yield json.dumps({
                    "type": "error",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }) + "\n"
            finally:
                # Session cleanup removed - using simpler approach
                pass

        return StreamingResponse(structured_agent_stream(), media_type="application/json")
        
    except Exception as e:
        async def error_stream():
            yield json.dumps({
                "type": "error",
                "error_type": type(e).__name__,
                "message": str(e)
            }) + "\n"
        return StreamingResponse(error_stream(), media_type="application/json")


@app.post("/mcp/clear-health-cache")
async def clear_health_cache():
    """Clear the server health cache to force fresh health checks."""
    global _server_health_cache
    _server_health_cache.clear()
    return {"message": "Health cache cleared"}

# --- Quick test endpoint for performance ---
@app.get("/api/quick-test")
async def quick_test():
    """Quick endpoint to test server response time without MCP checks."""
    return {
        "status": "ok", 
        "timestamp": time.time(),
        "message": "Server is responding quickly"
    }

#-----------Token Usage Endpoint for Monitoring and Debugging-----------
import tiktoken
@app.get("/chat/token-usage/{chat_id}")
def get_token_usage(chat_id: str):
    history = get_chat_history(chat_id)
    # history = chat_histories.get(chat_id, [])
    # Choose the encoding for your model, e.g., "cl100k_base" for GPT-3.5/4
    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    for msg in history:
        content = msg.get("content", "")
        total_tokens += len(enc.encode(content))
    return {"chat_id": chat_id, "total_tokens": total_tokens, "message_count": len(history)}

# @app.get("/llm/model-info")
# def get_llm_model_info():
#     # Try common attributes for model name
#     model_name = getattr(llm, "deployment_name", None) or getattr(llm, "model_name", None)
#     return {
#         "model_name": model_name,
#         "type": type(llm).__name__
#     }

@app.get("/llm/max-tokens")
def get_llm_max_tokens():
    # Try to get max tokens from the LLM object
    model_name = getattr(llm, "deployment_name", None) or getattr(llm, "model_name", None)

    return {"max_tokens": LLM_MAX_TOKENS.get(model_name, None), "model_name": model_name}


# Example: LLM model names and their max token limits
LLM_MAX_TOKENS = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-1106-preview": 128000,
    "gpt-4-turbo": 128000,
    "mistral-7b": 32768,
    "mistral-medium": 32768,
    "qwen-qwq-32b": 32768,
    "llama-2-70b": 4096,
    "llama-3-70b": 8192,
    "mixtral-8x7b": 32768,
    "claude-2": 100000,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "gpt-4o": 128000,
    # Add more as needed
}

# --- Playwright-compatible streaming endpoint ---
@app.post("/llm/playwright-structured-agent-stream")
async def playwright_structured_agent_stream(req: ChatRequest):
    """
    Streaming endpoint optimized for Playwright and stdio-based MCP servers.
    Uses individual client connections instead of MultiServerMCPClient for better compatibility.
    """
    servers = get_mcp_servers()
    if not servers:
        async def error_stream():
            yield json.dumps({"type": "error", "data": "No MCP servers configured."}) + "\n"
        return StreamingResponse(error_stream(), media_type="application/json")
    
    try:
        # Filter for reachable servers but handle stdio servers specially
        reachable_servers = await get_reachable_servers(servers, skip_health_check=True)
        if not reachable_servers:
            async def error_stream():
                yield json.dumps({"type": "error", "data": "No MCP servers configured."}) + "\n"
            return StreamingResponse(error_stream(), media_type="application/json")
        
        print(f"Using {len(reachable_servers)} servers: {list(reachable_servers.keys())}")
        
        # Create individual clients for each server (better for stdio servers like Playwright)
        all_tools = []
        active_clients = {}
        
        for server_name, server_config in reachable_servers.items():
            try:
                if server_config.get("transport") == "stdio":
                    # Handle stdio servers directly
                    from fastmcp.client.transports import StdioTransport
                    from fastmcp.client import Client
                    
                    transport = StdioTransport(
                        server_config["command"], 
                        server_config.get("args", [])
                    )
                    client = Client(transport)
                    await client.__aenter__()
                    active_clients[server_name] = client
                    
                    # Get tools from this client
                    try:
                        tools = await client.list_tools()
                        tools = [t.model_dump() if hasattr(t, 'model_dump') else t for t in tools]
                        
                        # Ensure proper tool schema format
                        for tool in tools:
                            if 'parameters' not in tool:
                                if 'inputSchema' in tool:
                                    tool['parameters'] = tool['inputSchema']
                                else:
                                    tool['parameters'] = {"type": "object", "properties": {}}
                        
                        print(f"[DEBUG] Tools loaded from {server_name}: {[t['name'] for t in tools]}")
                        all_tools.extend(tools)
                        
                    except Exception as e:
                        print(f"Error getting tools from {server_name}: {e}")
                        continue
                        
                else:
                    # Handle URL-based servers with MultiServerMCPClient
                    single_server = {server_name: server_config}
                    client = MultiServerMCPClient(single_server)
                    active_clients[server_name] = client
                    
                    try:
                        tools = await client.get_tools()
                        all_tools.extend(tools)
                        print(f"[DEBUG] Tools loaded from {server_name}: {[t.name for t in tools if hasattr(t, 'name')]}")
                    except Exception as e:
                        print(f"Error getting tools from {server_name}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error connecting to server {server_name}: {e}")
                continue
        
        if not all_tools:
            async def error_stream():
                yield json.dumps({"type": "error", "data": "No tools available from reachable MCP servers."}) + "\n"
            return StreamingResponse(error_stream(), media_type="application/json")
    finally:
        pass    
        # Create tool executor function that routes to appropriate client
    try:
        # --- Session-persistent Playwright MCP client ---
        # Use a global dict to store clients by chat_id
        import threading
        from fastmcp.client.transports import StdioTransport
        from fastmcp.client import Client
        from langchain_openai import AzureChatOpenAI
        from langgraph.prebuilt import create_react_agent
        # Thread-safe client store
        if not hasattr(playwright_structured_agent_stream, "_client_store"):
            playwright_structured_agent_stream._client_store = {}
            playwright_structured_agent_stream._client_lock = threading.Lock()
        client_store = playwright_structured_agent_stream._client_store
        client_lock = playwright_structured_agent_stream._client_lock

        # Find Playwright MCP server config
        playwright_server = None
        for name, cfg in get_mcp_servers().items():
            if cfg.get("transport") == "stdio":
                playwright_server = cfg
                break
        if not playwright_server:
            async def error_stream():
                yield json.dumps({"type": "error", "data": "No Playwright MCP stdio server configured."}) + "\n"
            return StreamingResponse(error_stream(), media_type="application/json")

        # Use chat_id for session persistence
        chat_id = getattr(req, "chat_id", None)
        client = None
        # Acquire lock for thread safety
        with client_lock:
            if chat_id and chat_id in client_store:
                client = client_store[chat_id]
            else:
                transport = StdioTransport(playwright_server["command"], playwright_server.get("args", []))
                client = Client(transport)
                # Enter async context outside lock
                client_store[chat_id] = client if chat_id else None

        # Enter async context if new client
        if not hasattr(client, "_entered") or not client._entered:
            await client.__aenter__()
            client._entered = True

        try:
            tools = await client.list_tools()
            tools = [t.model_dump() if hasattr(t, 'model_dump') else t for t in tools]
            for tool in tools:
                if 'parameters' not in tool:
                    if 'inputSchema' in tool:
                        tool['parameters'] = tool['inputSchema']
                    else:
                        tool['parameters'] = {"type": "object", "properties": {}}
        except Exception as e:
            # Clean up client on error
            with client_lock:
                if chat_id and chat_id in client_store:
                    del client_store[chat_id]
            await client.__aexit__(None, None, None)
            async def error_stream():
                yield json.dumps({"type": "error", "data": f"Could not list tools: {e}"}) + "\n"
            return StreamingResponse(error_stream(), media_type="application/json")

        llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            openai_api_type="azure",
        )
        agent = create_react_agent(llm, tools)

        # Prepare messages from history
        history = req.history
        if req.chat_id:
            history = get_and_update_chat_history(req.chat_id, req.history)
        recent_history = history[-10:] if history and len(history) > 10 else history
        messages = []
        if recent_history:
            for m in recent_history:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    messages.append(AIMessage(content=m["content"]))
        messages.append(HumanMessage(content=req.message))

        # Streaming agent-tool loop
        async def agent_stream():
            try:
                step_count = 0
                agent_messages = messages.copy()
                while True:
                    response = await agent.ainvoke({"messages": agent_messages})
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
                                yield json.dumps({
                                    "type": "final_answer",
                                    "content": msg.content
                                }) + "\n"
                        break
                    # Execute each tool and append results as function messages
                    for tc in tool_calls:
                        tool_name = tc.get('name')
                        tool_args = tc.get('args', {})
                        try:
                            result = await client.call_tool(tool_name, tool_args)
                            # Fix: ensure result is JSON serializable
                            if not isinstance(result, (str, int, float, bool, type(None), dict, list)):
                                result = str(result)
                        except Exception as e:
                            result = {"error": str(e)}
                        yield json.dumps({
                            "type": "tool_use",
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "result": result
                        }) + "\n"
                        # Add the tool result as a function message for next agent step
                        agent_messages.append({
                            "role": "function",
                            "name": tool_name,
                            "content": str(result)
                        })
                    await asyncio.sleep(0.01)
                yield json.dumps({"type": "stream_end"}) + "\n"
            except Exception as e:
                yield json.dumps({
                    "type": "error",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }) + "\n"
            finally:
                # Clean up client if requested (e.g., session end)
                if chat_id and getattr(req, "end_session", False):
                    with client_lock:
                        if chat_id in client_store:
                            del client_store[chat_id]
                    await client.__aexit__(None, None, None)

        return StreamingResponse(agent_stream(), media_type="application/json")
    except Exception as e:
        async def error_stream():
            yield json.dumps({
                "type": "error",
                "error_type": type(e).__name__,
                "message": str(e)
            }) + "\n"
        return StreamingResponse(error_stream(), media_type="application/json")

# --- Main entry point for running the FastAPI app ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cli1:app", host="0.0.0.0", port=8080, reload=True)