

import asyncio
from typing import List, Dict, Any, Optional, Union
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel

class StructuredAgent:
    def __init__(self, llm: BaseChatModel, tools: List[BaseTool], max_iterations: int = 30):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools} 
        self.max_iterations = max_iterations

    async def invoke(self, input: Union[str, List[BaseMessage]]) -> Dict[str, Any]:
        """Run the agent with input (text or messages)"""
        # Initialize state
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = input.copy()
        
        iterations = 0
        final_response = None
        reasoning_steps = []
        
        # Main loop
        while iterations < self.max_iterations:
            iterations += 1
            
            # Generate AI response (with tool binding)
            llm_with_tools = self.llm.bind_tools(list(self.tools.values()))
            ai_message = await llm_with_tools.ainvoke(messages)
            messages.append(ai_message)
            
            # Process the AI response with reasoning
            step_info = self._process_ai_response(ai_message, iterations)
            reasoning_steps.append(step_info)
            
            # Extract tool calls if any
            if not hasattr(ai_message, 'tool_calls') or not ai_message.tool_calls:
                final_response = ai_message.content
                break
                
            # Execute all tool calls and update the step info
            tool_messages = await self._execute_tools(ai_message.tool_calls)
            messages.extend(tool_messages)
            
            # Update step with tool results
            step_info['tool_results'] = self._format_tool_results(ai_message.tool_calls, tool_messages)
        
        return {
            "response": final_response or f"Stopped after {iterations} iterations",
            "reasoning_steps": reasoning_steps,
            "messages": messages,
            "iterations": iterations,
            "formatted_output": self._format_structured_response(reasoning_steps, final_response)
        }

    async def astream(self, input: Union[str, List[BaseMessage]]):
        """Stream the agent execution, yielding intermediate steps with structured formatting"""
        import json
        
        # Initialize state
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = input.copy()
        
        iterations = 0
        reasoning_steps = []
        
        # Main loop with streaming
        while iterations < self.max_iterations:
            iterations += 1
            
            # Generate AI response (with tool binding)
            llm_with_tools = self.llm.bind_tools(list(self.tools.values()))
            ai_message = await llm_with_tools.ainvoke(messages)
            messages.append(ai_message)
            
            # Process the AI response with reasoning
            step_info = self._process_ai_response(ai_message, iterations)
            reasoning_steps.append(step_info)
            
            # Yield the thinking step if there's reasoning content using structured tags
            if step_info['reasoning'] and step_info['reasoning'].strip():
                yield {
                    "type": "thought",
                    "iteration": iterations,
                    "content": step_info['reasoning'].strip()
                }
            
            # Extract tool calls if any
            if not hasattr(ai_message, 'tool_calls') or not ai_message.tool_calls:
                # No more tool calls, yield final response
                final_response = ai_message.content
                yield {
                    "type": "final_answer",
                    "content": final_response,
                    "iterations": iterations
                }
                break
                
            # Execute all tool calls and update the step info
            tool_messages = await self._execute_tools(ai_message.tool_calls)
            messages.extend(tool_messages)
            
            # Update step with tool results and yield structured tool executions
            tool_results = self._format_tool_results(ai_message.tool_calls, tool_messages)
            step_info['tool_results'] = tool_results
            
            # Use structured formatting for tool executions
            for tool_call in step_info['tool_calls']:
                tool_name = tool_call['name']
                args = tool_call['args']
                args_json = json.dumps(args, sort_keys=True, ensure_ascii=False)
                
                # Find corresponding result
                result = None
                for tool_result in tool_results:
                    if (tool_result['tool_name'] == tool_name and 
                        json.dumps(tool_result['arguments'], sort_keys=True, ensure_ascii=False) == args_json):
                        result = tool_result['result']
                        break
                
                # Yield action
                yield {
                    "type": "tool_execution",
                    "iteration": iterations,
                    "tool_name": tool_name,
                    "arguments": args,
                    "result": result
                }
        
        # If we exit the loop without a final answer, yield a timeout message
        if iterations >= self.max_iterations:
            yield {
                "type": "final_answer", 
                "content": f"Stopped after {iterations} iterations",
                "iterations": iterations
            }

    def _process_ai_response(self, ai_message: AIMessage, iteration: int) -> Dict[str, Any]:
        """Process AI response and extract reasoning + tool calls"""
        step_info = {
            'iteration': iteration,
            'reasoning': ai_message.content,
            'tool_calls': [],
            'tool_results': []
        }
        
        if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
            for tool_call in ai_message.tool_calls:
                step_info['tool_calls'].append({
                    'name': tool_call['name'],
                    'args': tool_call['args'],
                    'id': tool_call['id']
                })
        
        return step_info

    async def _execute_tools(self, tool_calls: List[Dict]) -> List[ToolMessage]:
        """Execute multiple tool calls asynchronously"""
        import traceback
        tool_messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            if tool_name not in self.tools:
                result = f"Error: Unknown tool '{tool_name}'"
            else:
                tool = self.tools[tool_name]
                try:
                    if asyncio.iscoroutinefunction(tool.arun):
                        result = await tool.arun(tool_args)
                    elif asyncio.iscoroutinefunction(tool.run):
                        result = await tool.run(tool_args)
                    else:
                        result = await asyncio.to_thread(tool.run, tool_args)
                except Exception as e:
                    # Enhanced error logging for debugging
                    error_msg = str(e) if str(e) else "Unknown error (empty exception message)"
                    print(f"Tool execution error for {tool_name}:")
                    print(f"  Arguments: {tool_args}")
                    print(f"  Result: {error_msg}")
                    print(f"  Error details: {e}")
                    print(f"  Full step: {{'type': 'tool_execution', 'iteration': 1, 'tool_name': '{tool_name}', 'arguments': {tool_args}, 'result': 'Error executing {tool_name}: {error_msg}'}}")
                    print(f"  Exception type: {type(e).__name__}")
                    print(f"  Traceback: {traceback.format_exc()}")
                    result = f"Error executing {tool_name}: {error_msg}"
            
            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call['id'],
                    name=tool_name
                )
            )
        
        return tool_messages

    def _format_tool_results(self, tool_calls: List[Dict], tool_messages: List[ToolMessage]) -> List[Dict]:
        """Format tool results with their corresponding calls"""
        results = []
        
        for tool_call in tool_calls:
            # Find corresponding result
            result_content = None
            for tool_msg in tool_messages:
                if tool_msg.tool_call_id == tool_call['id']:
                    result_content = tool_msg.content
                    break
            
            results.append({
                'tool_name': tool_call['name'],
                'arguments': tool_call['args'],
                'result': result_content
            })
        
        return results

    def _format_structured_response(self, reasoning_steps: List[Dict], final_response: str) -> str:
        """Format the response with tool results in tags for frontend parsing/rendering."""
        import json
        formatted_parts = []


        for step in reasoning_steps:
            # Add reasoning if present
            if step['reasoning'] and step['reasoning'].strip():
                formatted_parts.append(step['reasoning'].strip())

            # Always emit tool_call tags for every tool call, even if not executed
            # Map tool_results by tool_name+args for lookup
            tool_results_map = {}
            if step.get('tool_results'):
                for tool_result in step['tool_results']:
                    # Always use JSON string for args for matching
                    args_json = json.dumps(tool_result['arguments'], sort_keys=True, ensure_ascii=False)
                    key = (tool_result['tool_name'], args_json)
                    tool_results_map[key] = tool_result['result']

            if step.get('tool_calls'):
                for tool_call in step['tool_calls']:
                    tool_name = tool_call['name']
                    args = tool_call['args']
                    args_json = json.dumps(args, sort_keys=True, ensure_ascii=False)
                    key = (tool_name, args_json)
                    result = tool_results_map.get(key)
                    formatted_parts.append("<tool_call>")
                    formatted_parts.append(f"<tool_name>{tool_name}</tool_name>")
                    formatted_parts.append(f"<args>{args_json}</args>")
                    # Always tag the result, even if empty or None
                    if result is not None:
                        formatted_parts.append(f"<tool_result>{result}</tool_result>")
                    else:
                        formatted_parts.append(f"<tool_result></tool_result>")
                    formatted_parts.append("</tool_call>")

        # Add final response if present and not already included
        if final_response and final_response.strip():
            formatted_parts.append(final_response.strip())

        return '\n'.join(formatted_parts)

    def _extract_tool_executions(self, messages: List[BaseMessage]) -> List[Dict]:
        """Extract tool execution details from message history (legacy method)"""
        executions = []
        
        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls'):
                for tool_call in msg.tool_calls:
                    execution = {
                        'tool_name': tool_call['name'],
                        'arguments': tool_call['args'],
                        'tool_call_id': tool_call['id']
                    }
                    # Find corresponding tool result
                    for tool_msg in messages:
                        if (isinstance(tool_msg, ToolMessage) and 
                            tool_msg.tool_call_id == tool_call['id']):
                            execution['result'] = tool_msg.content
                            break
                    executions.append(execution)
        return executions

    def get_conversation_summary(self) -> str:
        """Get a summary of the last conversation in a readable format"""
        # This would be called after invoke() to get a nice summary
        pass