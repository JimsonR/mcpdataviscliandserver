# Structured JSON Streaming Protocol

## Overview

The `/llm/structured-agent-stream` endpoint now uses a robust structured JSON streaming protocol instead of XML-like tags embedded in strings. This makes frontend rendering much more reliable and maintainable.

## Stream Format

Each line in the stream is a JSON object followed by a newline character (`\n`). The JSON objects have different `type` values to indicate their purpose.

## Message Types

### 1. Thought Block
```json
{
  "type": "thought",
  "content": "### Question 3: What is the trend...",
  "step": 1
}
```
- **Purpose**: Agent reasoning and thinking process
- **Frontend**: Render `content` as markdown
- **Fields**:
  - `content`: Raw markdown content for rendering
  - `step`: Sequential step number

### 2. Tool Use Block
```json
### `tool_use` Messages

Tool execution blocks with enhanced formatting for better frontend display:

```json
{
  "type": "tool_use",
  "tool_name": "mcp_azdbexp_run_sql_query_enhanced",
  "arguments": {
    "sql": "SELECT TOP 10 * FROM customers",
    "purpose": "Get sample customer data"
  },
  "result": "Found 10 customers: [customer data...]",
  "is_parallel": false,
  "status": "completed",
  "execution_time": 1.23,
  "timestamp": 1643723400.123,
  "result_type": "success"
}
```

**Enhanced Tool Fields:**
- `tool_name`: Clean tool name (removes "functions." prefix)
- `arguments`: Formatted arguments (parsed JSON if string)
- `result`: Formatted result (truncated if >2000 chars)
- `status`: "executing" or "completed"
- `execution_time`: Time taken in seconds (if available)
- `timestamp`: Unix timestamp
- `result_type`: "success" or "error" (based on result content)
- `is_parallel`: Boolean indicating parallel execution
- `parallel_index`: Index for parallel tool calls
```
- **Purpose**: Tool execution details
- **Frontend**: Render as structured tool call with collapsible sections
- **Fields**:
  - `tool_name`: Name of the executed tool
  - `arguments`: Tool input parameters (JSON object)
  - `result`: Tool execution result (string or null if not yet available)
  - `is_parallel`: Boolean indicating if this is part of parallel execution
  - `parallel_index`: Index for parallel tool calls (optional)

### 3. Final Answer Block
```json
{
  "type": "final_answer",
  "content": "Sales Trend Over the Last Three Years..."
}
```
- **Purpose**: Final response/conclusion
- **Frontend**: Render `content` as markdown, typically with special styling
- **Fields**:
  - `content`: Final answer content in markdown format

### 4. Error Block
```json
{
  "type": "error",
  "error_type": "ValueError",
  "message": "Database connection failed"
}
```
- **Purpose**: Error reporting
- **Frontend**: Display error message to user
- **Fields**:
  - `error_type`: Type of exception that occurred
  - `message`: Human-readable error message

### 5. Stream End Block
```json
{
  "type": "stream_end"
}
```
- **Purpose**: Indicates successful completion of the stream
- **Frontend**: Hide loading indicators, finalize UI

## Frontend Implementation Example

```javascript
// Parse each line as JSON
const lines = streamData.split('\n').filter(line => line.trim());

for (const line of lines) {
  try {
    const message = JSON.parse(line);
    
    switch (message.type) {
      case 'thought':
        renderMarkdown(message.content, 'thinking-block');
        break;
        
      case 'tool_use':
        renderToolBlock({
          name: message.tool_name,
          input: message.arguments,
          output: message.result
        });
        break;
        
      case 'final_answer':
        renderMarkdown(message.content, 'final-answer-block');
        break;
        
      case 'error':
        showError(`${message.error_type}: ${message.message}`);
        break;
        
      case 'stream_end':
        hideLoadingIndicator();
        break;
    }
  } catch (e) {
    console.error('Failed to parse stream message:', line);
  }
}
```

## Enhanced Frontend Implementation

### Tool Block Component Example
```jsx
const ToolBlock = ({ toolName, arguments, result, status, executionTime, resultType, isParallel }) => {
  return (
    <div className={`tool-block ${resultType} ${status}`}>
      <div className="tool-header">
        <div className="tool-info">
          <span className="tool-name">{toolName}</span>
          {isParallel && <span className="parallel-badge">Parallel</span>}
        </div>
        <div className="tool-status">
          <span className={`status-indicator ${status}`}>
            {status === 'executing' ? '⏳' : status === 'completed' ? '✅' : '❌'}
          </span>
          {executionTime && <span className="timing">{executionTime.toFixed(2)}s</span>}
        </div>
      </div>
      
      <div className="tool-content">
        <details className="arguments-section">
          <summary>Arguments</summary>
          <pre className="json-display">{JSON.stringify(arguments, null, 2)}</pre>
        </details>
        
        {result && (
          <div className="result-section">
            <h4>Result</h4>
            <div className={`result-content ${resultType}`}>
              <pre>{result}</pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
```

### CSS Styling for Enhanced Tool Blocks
```css
.tool-block {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  margin: 12px 0;
  overflow: hidden;
  transition: all 0.2s ease;
}

.tool-block.success {
  border-left: 4px solid #4caf50;
}

.tool-block.error {
  border-left: 4px solid #f44336;
}

.tool-block.executing {
  border-left: 4px solid #ff9800;
  animation: pulse 2s infinite;
}

.tool-header {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  padding: 12px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #dee2e6;
}

.tool-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.tool-name {
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-weight: 600;
  color: #495057;
}

.parallel-badge {
  background: #6f42c1;
  color: white;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.75em;
  font-weight: 500;
}

.tool-status {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-indicator {
  font-size: 1.2em;
}

.timing {
  color: #6c757d;
  font-size: 0.85em;
  font-family: monospace;
}

.tool-content {
  padding: 16px;
}

.arguments-section summary {
  cursor: pointer;
  font-weight: 500;
  margin-bottom: 8px;
  color: #495057;
}

.json-display {
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 4px;
  padding: 12px;
  overflow-x: auto;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 0.9em;
  line-height: 1.4;
}

.result-section h4 {
  margin: 16px 0 8px 0;
  color: #495057;
  font-size: 0.95em;
  font-weight: 600;
}

.result-content {
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 4px;
  padding: 12px;
  max-height: 300px;
  overflow-y: auto;
}

.result-content.success {
  border-color: #d4edda;
  background-color: #d1ecf1;
}

.result-content.error {
  border-color: #f5c6cb;
  background-color: #f8d7da;
}

.result-content pre {
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 0.9em;
  line-height: 1.4;
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.7; }
  100% { opacity: 1; }
}
```

## Benefits

1. **No XML Parsing**: Frontend doesn't need to parse XML-like tags from strings
2. **Type Safety**: Clear message types for robust handling
3. **Structured Data**: Tool arguments and results are proper JSON objects
4. **Error Handling**: Structured error reporting
5. **Markdown Ready**: Content is ready for markdown rendering
6. **Extensible**: Easy to add new message types without breaking existing code

## Migration Notes

- Replace any existing XML tag parsing with type-based message handling
- Use a markdown renderer (like `marked.js` or `react-markdown`) for `content` fields
- Handle `tool_use` messages with structured display of inputs/outputs
- Implement proper error states for `error` messages
