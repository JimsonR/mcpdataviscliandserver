# MCP Data Visualization Backend

A comprehensive FastAPI backend with Model Context Protocol (MCP) integration for data exploration, visualization, and AI-powered analytics. This project combines multiple MCP servers (stdio and HTTP) with streaming AI agents for interactive data analysis.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   MCP Servers   │
│   Client        │◄──►│   Backend       │◄──►│   (stdio/HTTP)  │
│                 │    │   (cli1.py)     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Azure SQL     │
                       │   Database      │
                       └─────────────────┘
```

## 📦 Components

### 1. FastAPI Backend (`cli1.py`)
Main server providing:
- **Multi-server MCP integration** (stdio + HTTP transports)
- **Streaming AI agents** with structured responses
- **Session persistence** for chat history and MCP clients
- **Process management** to prevent Node.js process accumulation
- **RESTful APIs** for data operations

### 2. Azure SQL MCP Server (`azdbexp.py`)
Specialized MCP server for:
- **Azure SQL database operations**
- **Data exploration and analysis**
- **Automated insights generation**
- **Hypothesis testing**
- **Data visualization preparation**

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js (for Playwright MCP server)
- Azure SQL Database access
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd mcpdatavisbackend
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
Create a `.env` file:
```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Azure SQL Configuration
AZURE_SQL_SERVER=your-server.database.windows.net
AZURE_SQL_DATABASE=your_database
AZURE_SQL_USERNAME=your_username
AZURE_SQL_PASSWORD=your_password
AZURE_SQL_DRIVER={ODBC Driver 18 for SQL Server}

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379
```

4. **Configure MCP servers**
Edit `mcp_servers.yaml`:
```yaml
servers:
  azdbexp:
    transport: streamable_http
    url: http://localhost:8005/mcp
  playwright:
    args:
    - "@playwright/mcp@latest"
    - --browser
    - chrome
    - --isolated
    command: npx
    transport: stdio
```

5. **Start the servers**

Start Azure SQL MCP server:
```bash
python azdbexp.py
```

Start main FastAPI backend:
```bash
python cli1.py
```

## 📚 API Documentation

### Core Endpoints

#### Streaming Agent Endpoints

##### Multi-Server Structured Agent Stream
```http
POST /llm/structured-agent-stream
Content-Type: application/json

{
  "message": "Analyze sales data by territory",
  "chat_id": "optional-session-id",
  "history": []
}
```

**Response Format:**
```json
{"type": "tool_use", "tool_name": "run_sql_query_enhanced", "status": "executing", "server": "azdbexp", ...}
{"type": "tool_use", "tool_name": "run_sql_query_enhanced", "status": "completed", "result": "...", ...}
{"type": "final_answer", "content": "Analysis results..."}
{"type": "stream_end"}
```

##### Legacy Structured Agent Stream
```http
POST /llm/llm-structured-agent-stream
```
Uses only HTTP-based MCP servers with MultiServerMCPClient.

#### MCP Server Management

##### List Available Tools
```http
GET /mcp/list-tools?server=azdbexp
```

##### Call MCP Tool
```http
POST /mcp/call-tool?server=azdbexp&tool_name=run_sql_query_enhanced
{
  "sql": "SELECT TOP 10 * FROM sales_data",
  "purpose": "Data exploration"
}
```

##### Get Active Processes
```http
GET /mcp/active-processes
```

#### Chat Session Management

##### Create Session
```http
POST /chat/create-session
```

##### Get Chat History
```http
GET /chat/get-history/{chat_id}
```

##### List Sessions
```http
GET /chat/list-sessions
```

##### Delete Session
```http
DELETE /chat/delete-session/{chat_id}
```

#### Health & Monitoring

##### Server Health Check
```http
GET /api/quick-test
```

##### Clear Health Cache
```http
POST /mcp/clear-health-cache
```

##### Token Usage
```http
GET /chat/token-usage/{chat_id}
```

## 🛠️ Azure SQL MCP Server (`azdbexp.py`)

### Available Tools

#### Database Operations
- **`run_sql_query_enhanced`**: Execute SQL queries with exploration tracking
- **`list_db_tables`**: List all database tables and their schemas
- **`restart_azure_sql_connection`**: Restart database connection

#### Data Exploration
- **`explore_data_patterns`**: Systematic data exploration (quality, relationships, patterns, anomalies)
- **`test_hypothesis`**: Test data hypotheses with validation
- **`get_exploration_history`**: Get complete exploration history

#### Data Management
- **`run_script`**: Execute Python scripts on loaded DataFrames
- **`list_dataframes`**: List all DataFrames in memory
- **`get_notes`**: Get exploration notes

#### Visualization
- **`create_visualization`**: Generate chart data for frontend
- **`list_supported_chart_types`**: List available chart types
- **`prepare_table_resource`**: Prepare table previews
- **`export_table_csv`**: Export data to CSV

### Supported Chart Types
- Histogram
- Line chart
- Bar chart
- Stacked bar chart
- Pie chart
- Area chart
- Scatter plot
- Heatmap
- Box plot

### Example Usage

#### SQL Query with Exploration
```python
# Tool: run_sql_query_enhanced
{
  "sql": "SELECT Territory, SUM(SalesAmount) as TotalSales FROM Sales GROUP BY Territory ORDER BY TotalSales DESC",
  "df_name": "territory_sales",
  "purpose": "Analyze sales by territory",
  "hypothesis": "Some territories significantly outperform others"
}
```

#### Data Visualization
```python
# Tool: create_visualization
{
  "df_name": "territory_sales",
  "plot_type": "bar",
  "x": "Territory",
  "y": "TotalSales",
  "title": "Sales by Territory"
}
```

#### Data Exploration
```python
# Tool: explore_data_patterns
{
  "df_name": "territory_sales",
  "exploration_type": "quality",
  "specific_columns": ["Territory", "TotalSales"]
}
```

## 🔧 Configuration

### MCP Server Configuration (`mcp_servers.yaml`)

```yaml
servers:
  # HTTP-based server
  azdbexp:
    transport: streamable_http
    url: http://localhost:8005/mcp
    
  # stdio-based server  
  playwright:
    args:
    - "@playwright/mcp@latest"
    - --browser
    - chrome
    - --isolated
    command: npx
    transport: stdio
    
  # Additional servers...
  simple-arxiv:
    args:
    - -m
    - mcp_simple_arxiv
    command: python
    transport: stdio
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Yes |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | Yes |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Model deployment name | Yes |
| `AZURE_OPENAI_API_VERSION` | API version | Yes |
| `AZURE_SQL_SERVER` | Azure SQL server name | Yes |
| `AZURE_SQL_DATABASE` | Database name | Yes |
| `AZURE_SQL_USERNAME` | Database username | Yes |
| `AZURE_SQL_PASSWORD` | Database password | Yes |
| `AZURE_SQL_DRIVER` | ODBC driver name | Yes |
| `REDIS_URL` | Redis connection URL | No |

## 🔄 Process Management

The system includes intelligent process management to prevent Node.js process accumulation:

### Session-Persistent Clients
- **Global client store**: Reuses MCP clients across requests
- **Session-based cleanup**: Clients tied to chat sessions
- **Automatic cleanup**: Processes cleaned up on session end or server restart

### Process Monitoring
- **Active process tracking**: Monitor stdio processes (Node.js, Python)
- **Health checks**: Validate client connections
- **Cleanup mechanisms**: Automatic and manual cleanup options

### Best Practices
- Use `chat_id` for session persistence
- Call `/mcp/active-processes` to monitor process count
- Use `end_session: true` in requests to trigger cleanup
- Restart server periodically to clear accumulated processes

## 📊 Response Formats

### Structured Streaming Response
All streaming endpoints follow a consistent JSON format:

```json
// Tool execution start
{
  "type": "tool_use",
  "tool_name": "run_sql_query_enhanced",
  "server": "azdbexp",
  "arguments": {"sql": "SELECT * FROM sales"},
  "status": "executing",
  "timestamp": 1693747200.0
}

// Tool execution complete
{
  "type": "tool_use",
  "tool_name": "run_sql_query_enhanced",
  "server": "azdbexp",
  "arguments": {"sql": "SELECT * FROM sales"},
  "result": "Query executed successfully...",
  "status": "completed",
  "execution_time": 1.5,
  "result_type": "success"
}

// Final answer
{
  "type": "final_answer",
  "content": "Based on the analysis..."
}

// Stream end
{
  "type": "stream_end"
}

// Error (if any)
{
  "type": "error",
  "error_type": "ValidationError",
  "message": "Invalid SQL syntax"
}
```

## 🚨 Error Handling

### Common Issues

#### Schema Validation Errors
```json
{
  "error": {
    "message": "Invalid schema for function 'tool_name': schema must be a JSON Schema of 'type: \"object\"', got 'type: \"None\"'."
  }
}
```
**Solution**: Schema validation is automatically handled in the updated endpoints.

#### Process Accumulation
**Symptoms**: Multiple Node.js processes running
**Solution**: Use session persistence and proper cleanup mechanisms implemented in the system.

#### Connection Issues
**Symptoms**: "No MCP servers are currently reachable"
**Solution**: 
1. Check if MCP servers are running
2. Verify `mcp_servers.yaml` configuration
3. Clear health cache: `POST /mcp/clear-health-cache`

## 🧪 Testing

### Manual Testing

1. **Start all servers**
```bash
# Terminal 1: Azure SQL MCP server
python azdbexp.py

# Terminal 2: Main FastAPI backend
python cli1.py
```

2. **Test health endpoint**
```bash
curl http://localhost:8080/api/quick-test
```

3. **Test MCP servers**
```bash
curl "http://localhost:8080/mcp/list-tools?server=azdbexp"
```

4. **Test streaming endpoint**
```bash
curl -X POST http://localhost:8080/llm/structured-agent-stream \
  -H "Content-Type: application/json" \
  -d '{"message": "List all database tables"}'
```

### Example Queries

#### Data Analysis
```json
{
  "message": "Show me the top 10 territories by sales amount",
  "chat_id": "analysis_session_1"
}
```

#### Data Visualization
```json
{
  "message": "Create a bar chart showing sales by product category",
  "chat_id": "viz_session_1"
}
```

#### Hypothesis Testing
```json
{
  "message": "Test if there's a correlation between territory size and sales performance",
  "chat_id": "research_session_1"
}
```

## 📈 Performance Considerations

### Optimization Features
- **Health caching**: 60-second cache for server health checks
- **Tool schema validation**: Prevents invalid tool calls
- **Process reuse**: Session-persistent MCP clients
- **Memory management**: Automatic DataFrame cleanup
- **Query optimization**: Built-in query result limiting

### Scaling Considerations
- **Horizontal scaling**: Multiple FastAPI instances with shared Redis
- **Database optimization**: Connection pooling and query optimization
- **Resource limits**: Configurable max points for visualizations
- **Caching strategies**: Redis for session data and health checks

## 🔐 Security

### Authentication
- Environment variable-based configuration
- Azure SQL integrated authentication support
- API key protection for Azure OpenAI

### Data Protection
- No sensitive data logging
- Secure connection strings
- Session-based data isolation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request with detailed description

## 📄 License

[Add your license information here]

## 🆘 Support

For issues and questions:
1. Check the error handling section above
2. Review server logs for detailed error messages
3. Use the health check endpoints to diagnose issues
4. Create an issue in the repository with detailed information

---

**Note**: This project is designed for data exploration and analytics workflows. Ensure proper database permissions and resource limits are configured for production use.
