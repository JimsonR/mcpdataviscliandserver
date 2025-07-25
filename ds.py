
from enum import Enum
import os
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, Prompt, PromptArgument, Resource  
import pandas as pd
import numpy as np
import json
import sys
import io
import traceback
import inspect
import re
from typing import Optional, List ,Dict
from pydantic import BaseModel

# Try to import optional libraries for out-of-core processing
try:
    import dask.dataframe as dd
except ImportError:
    dd = None

try:
    import polars as pl
except ImportError:
    pl = None

mcp = FastMCP(name="mcp_server_ds", host="127.0.0.1", port=8003)

# In-memory data store for loaded DataFrames
_dataframes = {}
_df_count = 0
_notes: list[str] = []

def _next_df_name():
    global _df_count
    _df_count += 1
    return f"df_{_df_count}"

class LoadCsvArgs(BaseModel):
    csv_path: str
    df_name: Optional[str] = None

class RunScriptArgs(BaseModel):
    script: str
    save_to_memory: Optional[List[str]] = None

@mcp.tool()
def load_csv(args: LoadCsvArgs) -> list:
    """Load a local CSV file into a DataFrame."""
    global _dataframes, _notes
    csv_path = args.csv_path
    df_name = args.df_name
    if not df_name:
        df_name = _next_df_name()
    
    try:
        _dataframes[df_name] = pd.read_csv(csv_path, encoding='latin1')
        _notes.append(f"Successfully loaded CSV into dataframe '{df_name}'")
        return [TextContent(type="text", text=f"Successfully loaded CSV into dataframe '{df_name}'")]
    except Exception as e:
        error_msg = f"Error loading CSV: {str(e)}"
        _notes.append(error_msg)
        return [TextContent(type="text", text=error_msg)]

#---------------------------------------------------------------------------------------------#
    # BIG DATA TEST 
#---------------------------------------------------------------------------------------------#
# Define the arguments for loading Parquet files
class LoadParquetArgs(BaseModel):
    parquet_path: str
    df_name: Optional[str] = None

@mcp.tool()
async def load_parquet(args : LoadParquetArgs) -> list:
    """Load a local parquet file into a DataFrame."""
    global _dataframes, _notes
    args = args
    parquet_path = args.parquet_path
    df_name = args.df_name
    if not df_name:
        df_name = _next_df_name()
    
    try:
        _dataframes[df_name] = pd.read_parquet(rf"{parquet_path}")
        _notes.append(f"Successfully loaded Parquet into dataframe '{df_name}'")
        return [TextContent(type="text", text=f"Successfully loaded Parquet into dataframe '{df_name}'")]
    except Exception as e:
        error_msg = f"Error loading Parquet: {str(e)}"
        _notes.append(error_msg)
        return [TextContent(type="text", text=error_msg)]

BIG_DATA_PROMPT_TEMPLATE= """
You are a professional Data Scientist. Perform exploratory data analysis on the dataset at this path:

<parquet_path>
{parquet_path}
</parquet_path>

Focus your analysis on: **{topic}**

## Your Task:
1. Load and explore the dataset structure
2. Identify 3-5 key insights related to {topic}
3. Create appropriate visualizations to support your findings
4. Provide a concise summary of your analysis

## Important Guidelines:
- Always call `list_dataframes` before using `run_script` or `create_visualization`
- Use `list_supported_chart_types` to discover available chart types before `create_visualization`
- Keep individual script outputs manageable (limit large results)
- Focus on the most relevant insights for the given topic
- Use visualizations to enhance understanding

Begin by loading the Parquet file and exploring its basic structure.
"""
@mcp.prompt()

def big_data_prompt(parquet_path: str, topic = "General data exploration") -> str:
    """
    A prompt to explore a parquet file and generate insights.
    """
    return BIG_DATA_PROMPT_TEMPLATE.format(
        parquet_path=parquet_path,
        topic=topic
    )

#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------#
    # API CALL TEST 
#---------------------------------------------------------------------------------------------#

class ApiCallArgs(BaseModel):
    api_url: str
    params: Optional[Dict[str, str]] = None
    headers: Optional[Dict[str, str]] = None
    auth: Optional[str] = f"Bearer {os.getenv('NASDAQ_API_KEY')}"
    request_type: Optional[str] = "GET"  # Default to GET, can be POST, PUT, etc.
    request_body: Optional[Dict[str, str]] = None  # For POST/PUT requests
    # api_key: Optional[str] = os.getenv("NASDAQ_API_KEY")  # Optional API key for authentication

import httpx

@mcp.tool()
async def call_api(args: ApiCallArgs) -> list:
    """Make an API call to the specified URL with optional parameters, headers, and request body."""
    api_url = args.api_url
    params = args.params or {}
    headers = args.headers or {}
    headers["Content-Type"] = "application/json"  # Default to JSON content type
    
    headers["Authorization"] = f"Bearer {os.getenv('NASDAQ_API_KEY')}" if "NASDAQ_API_KEY" in os.environ else headers.get("Authorization", "")
    request_type = args.request_type.upper()
    request_body = args.request_body or {}

    try:
        async with httpx.AsyncClient() as client:
            if request_type == "GET":
                response = await client.get(api_url, params=params, headers=headers)
            elif request_type == "POST":
                response = await client.post(api_url, json=request_body, params=params, headers=headers)
            elif request_type == "PUT":
                response = await client.put(api_url, json=request_body, params=params, headers=headers)
            elif request_type == "DELETE":
                response = await client.delete(api_url, params=params, headers=headers)
            else:
                return [TextContent(type="text", text=f"Unsupported request type: {request_type}")]

            response.raise_for_status()  # Raise an error for bad responses
            return [TextContent(type="text", text=response.text)]
    except httpx.RequestError as e:
        return [TextContent(type="text", text=f"Request error: {str(e)}")]
    except httpx.HTTPStatusError as e:
        return [TextContent(type="text", text=f"HTTP error: {str(e)}")]

#---------------------------------------------------------------------------------------------#
    
#---------------------------------------------------------------------------------------------#

@mcp.tool()
def get_notes() -> list:
    """Return the notes generated by the data exploration server."""
    global _notes
    return [TextContent(type="text", text="\n".join(_notes))]
@mcp.tool()
def run_script(args: RunScriptArgs) -> list:
    """Execute a Python script for data analytics tasks.
    Do list_dataframes before using this tool to ensure you know the available DataFrames and their columns.
    This tool allows you to run scripts that can analyze, process, and visualize data using pandas."""
    global _dataframes, _notes
    if not hasattr(run_script, "_memory"):
        run_script._memory = {}
    memory = run_script._memory
    script = args.script
    save_to_memory = args.save_to_memory
    
    # Basic validation
    if 'pd.read_csv(' in script:
        return [TextContent(type="text", text="ERROR: Use list_dataframes to check the available dataframes instead of pd.read_csv()")]
    if any(pattern in script for pattern in ['import matplotlib', 'plt.show(', 'plt.figure(']):
        return [TextContent(type="text", text="ERROR: Use DataFrame.plot() methods instead of matplotlib")]
    
    # If script is a single variable name, try to return from memory
    if isinstance(script, str) and script.strip() in memory and (not ("\n" in script or ";" in script)):
        val = memory[script.strip()]
        return [TextContent(type="text", text=repr(val))]
    
    # Import required libraries
    import pandas as pd
    import numpy as np
    import sys
    import io
    import traceback
    
    # Prepare execution context
    local_vars = {**_dataframes, **memory}
    
    # Initialize variables that will be used in finally block
    stdout = io.StringIO()
    sys_stdout = sys.stdout  # Store original stdout before any try block
    result = None
    
    try:
        # Split script into lines, try to eval last line if possible
        lines = script.strip().split("\n")
        # Remove empty lines at end
        while lines and not lines[-1].strip():
            lines.pop()
        last_line = lines[-1] if lines else ""
        body = "\n".join(lines[:-1])
        
        # Prepare globals for exec/eval: include builtins, pd, np
        exec_globals = globals().copy()
        exec_globals.update({'pd': pd, 'np': np})
        
        # Redirect stdout
        sys.stdout = stdout
        
        # Execute all but last line
        if body.strip():
            exec(body, exec_globals, local_vars)
        
        # Try to eval last line
        try:
            result = eval(last_line, exec_globals, local_vars)
        except Exception:
            # If eval fails, exec last line
            exec(last_line, exec_globals, local_vars)
            result = None
        
        # Save to memory if requested
        # Check if save_to_memory was set in the script itself
        if 'save_to_memory' in local_vars and isinstance(local_vars['save_to_memory'], list):
            save_to_memory = local_vars['save_to_memory']
        
        saved_vars = []
        if save_to_memory:
            for idx, name in enumerate(save_to_memory):
                val = None
                if name in local_vars:
                    val = local_vars[name]
                elif name in locals():
                    val = locals()[name]
                # If not found, and this is the first name, and result is a DataFrame, save it
                elif idx == 0 and isinstance(result, pd.DataFrame):
                    val = result
                if val is not None:
                    memory[name] = val
                    saved_vars.append(name)
                    # If it's a DataFrame or Series, also add to _dataframes
                    if isinstance(val, (pd.DataFrame, pd.Series)):
                        _dataframes[name] = val
            if saved_vars:
                _notes.append(f"Saved to memory: {', '.join(saved_vars)}")
        
        # If result is None, but stdout has output, return that
        output_text = stdout.getvalue().strip()
        if result is not None:
            output = repr(result)
        elif output_text:
            output = output_text
        else:
            # Check if we saved anything and report it
            if save_to_memory and saved_vars:
                output = f"Script executed successfully. Saved to memory: {', '.join(saved_vars)}"
            else:
                output = "Script executed successfully (no output)"
            
    except Exception as e:
        tb = traceback.format_exc()
        output = f"Error: {str(e)}\n{tb}"
        
    finally:
        # Restore original stdout (sys_stdout is guaranteed to be defined)
        sys.stdout = sys_stdout
    
    _notes.append(f"Script executed: {output[:100]}...")
    return [TextContent(type="text", text=output)]
@mcp.tool()
def list_dataframes() -> list:
    """List all DataFrames currently loaded in memory."""
    global _dataframes
    if not _dataframes:
        return [TextContent(type="text", text="No DataFrames loaded. Use load_csv tool first.")]
    
    result = "=== DATAFRAMES IN MEMORY ===\n\n"
    for name, df in _dataframes.items():
        if isinstance(df, pd.DataFrame):
            result += f"DataFrame: {name}\n"
            result += f"  Type: DataFrame\n"
            result += f"  Shape: {df.shape}\n"
            result += f"  Columns: {list(df.columns)}\n\n"
        elif isinstance(df, pd.Series):
            result += f"Series: {name}\n"
            result += f"  Type: Series\n"
            result += f"  Shape: {df.shape}\n"
            result += f"  Dtype: {df.dtype}\n\n"
        else:
            result += f"Unknown type for: {name} ({type(df)})\n\n"
    result += f"Use these names in scripts: {', '.join(_dataframes.keys())}"
    return [TextContent(type="text", text=result)]


# --- Improved modular visualization tool ---
class CreateVisualizationArgs(BaseModel):
    df_name: str
    plot_type: str
    x: Optional[str] = None
    y: Optional[str] = None
    column: Optional[str] = None
    title: Optional[str] = None
    bins: int = 20
    max_points: int = 100
    max_categories: int = 20  # For area/stacked charts
    auto_limit: bool = True  # If True, fallback to top-N

# @mcp.tool()
def _extract_plot_data(df, plot_type, x=None, y=None, column=None, title=None, bins=20, max_points=100, max_processing_rows=500000):
    """
    Extract plot data from DataFrame with intelligent sampling for large datasets.
    
    Args:
        df: DataFrame to extract data from
        plot_type: Type of plot to generate
        x, y, column: Column names for plotting
        title: Optional title for the plot
        bins: Number of bins for histogram
        max_points: Maximum points in final output
        max_processing_rows: Maximum rows to process before sampling
    """
    global np
    import numpy as np
    import pandas as pd
    # Lower threshold for area/complex charts
    area_types = ["area", "heatmap"]
    # Use a lower threshold for area/complex charts
    if plot_type in area_types:
        effective_max_processing_rows = min(50000, max_processing_rows)
    else:
        effective_max_processing_rows = max_processing_rows

    # Early sampling for large datasets to prevent memory issues
    original_size = len(df)
    # Try to use Dask or Polars for large DataFrames and expensive charts
    dask_used = False
    polars_used = False
    dask_df = None
    polars_df = None
    # Only try for area/heatmap and large DataFrames
    if plot_type in area_types and original_size > effective_max_processing_rows:
        try:
            import dask.dataframe as dd
            dask_df = dd.from_pandas(df, npartitions=8)
            dask_used = True
        except ImportError:
            try:
                import polars as pl
                polars_df = pl.from_pandas(df)
                polars_used = True
            except ImportError:
                pass

    if dask_used:
        print(f"Using Dask for out-of-core processing: {original_size:,} rows.")
        # For area/heatmap, sample after groupby to avoid memory issues
        if plot_type == "heatmap" and x and y:
            if column:
                pivot = dask_df.pivot_table(index=y, columns=x, values=column, aggfunc="sum", fill_value=0).compute()
                title_default = f"Heatmap of {column} by {y} vs {x}"
            else:
                # Fallback to counting occurrences
                pivot = dask_df.groupby([y, x]).size().unstack(fill_value=0).compute()
                title_default = f"Heatmap count of {y} vs {x}"
            # Limit heatmap dimensions
            if len(pivot.columns) > max_points:
                pivot = pivot.iloc[:, :max_points]
            if len(pivot.index) > max_points:
                pivot = pivot.iloc[:max_points, :]
            return {
                "type": "heatmap",
                "x": list(pivot.columns),
                "y": list(pivot.index),
                "z": pivot.values.tolist(),
                "title": title or title_default,
                "aggregated_column": column if column else None,
                "sampled": True,
                "original_size": original_size,
                "out_of_core": "dask"
            }
        elif plot_type == "area" and x and y:
            # Only support split_column (series) for Dask path
            split_column = column if column and column in dask_df.columns else None
            if split_column:
                grouped = dask_df[[x, split_column, y]].dropna().groupby([x, split_column])[y].sum().reset_index().compute()
                area_series = {}
                for series_name in grouped[split_column].unique():
                    series_data = grouped[grouped[split_column] == series_name]
                    x_vals = series_data[x].tolist()
                    y_vals = series_data[y].tolist()
                    if len(x_vals) > max_points:
                        step = max(1, len(x_vals) // max_points)
                        x_vals = x_vals[::step]
                        y_vals = y_vals[::step]
                    area_series[str(series_name)] = [
                        {"x": str(xv), "y": float(yv)} for xv, yv in zip(x_vals, y_vals)
                    ]
                return {
                    "type": "area",
                    "series": area_series,
                    "title": title or f"Area chart of {y} by {x} per {split_column}",
                    "x": x,
                    "y": y,
                    "series_column": split_column,
                    "sampled": True,
                    "original_size": original_size,
                    "out_of_core": "dask"
                }
            else:
                # Fallback: treat y as a list of columns (legacy behavior)
                y_cols = y if isinstance(y, list) else [y]
                data = dask_df[[x] + y_cols].dropna().compute()
                x_vals = data[x].tolist()
                area_series = {}
                for y_col in y_cols:
                    y_vals = data[y_col].tolist()
                    if len(x_vals) > max_points:
                        step = max(1, len(x_vals) // max_points)
                        x_vals_ds = x_vals[::step]
                        y_vals_ds = y_vals[::step]
                    else:
                        x_vals_ds = x_vals
                        y_vals_ds = y_vals
                    area_series[y_col] = [{"x": str(xv), "y": float(yv)} for xv, yv in zip(x_vals_ds, y_vals_ds)]
                return {
                    "type": "area",
                    "series": area_series,
                    "title": title or f"Area chart of {y} by {x}",
                    "x": x,
                    "y": y,
                    "sampled": True,
                    "original_size": original_size,
                    "out_of_core": "dask"
                }
    elif polars_used:
        print(f"Using Polars for out-of-core processing: {original_size:,} rows.")
        import polars as pl
        if plot_type == "heatmap" and x and y:
            if column:
                pivot = polars_df.pivot(values=column, index=y, columns=x, aggregate_fn="sum").to_pandas()
                title_default = f"Heatmap of {column} by {y} vs {x}"
            else:
                # Fallback to counting occurrences
                temp = polars_df.groupby([y, x]).count().to_pandas()
                pivot = temp.pivot(index=y, columns=x, values="count")
                title_default = f"Heatmap count of {y} vs {x}"
            if len(pivot.columns) > max_points:
                pivot = pivot.iloc[:, :max_points]
            if len(pivot.index) > max_points:
                pivot = pivot.iloc[:max_points, :]
            return {
                "type": "heatmap",
                "x": list(pivot.columns),
                "y": list(pivot.index),
                "z": pivot.values.tolist(),
                "title": title or title_default,
                "aggregated_column": column if column else None,
                "sampled": True,
                "original_size": original_size,
                "out_of_core": "polars"
            }
        elif plot_type == "area" and x and y:
            split_column = column if column and column in polars_df.columns else None
            if split_column:
                grouped = polars_df.drop_nulls().groupby([x, split_column]).agg([pl.col(y).sum()]).to_pandas()
                area_series = {}
                for series_name in grouped[split_column].unique():
                    series_data = grouped[grouped[split_column] == series_name]
                    x_vals = series_data[x].tolist()
                    y_vals = series_data[y].tolist()
                    if len(x_vals) > max_points:
                        step = max(1, len(x_vals) // max_points)
                        x_vals = x_vals[::step]
                        y_vals = y_vals[::step]
                    area_series[str(series_name)] = [
                        {"x": str(xv), "y": float(yv)} for xv, yv in zip(x_vals, y_vals)
                    ]
                return {
                    "type": "area",
                    "series": area_series,
                    "title": title or f"Area chart of {y} by {x} per {split_column}",
                    "x": x,
                    "y": y,
                    "series_column": split_column,
                    "sampled": True,
                    "original_size": original_size,
                    "out_of_core": "polars"
                }
            else:
                y_cols = y if isinstance(y, list) else [y]
                data = polars_df.select([x] + y_cols).drop_nulls().to_pandas()
                x_vals = data[x].tolist()
                area_series = {}
                for y_col in y_cols:
                    y_vals = data[y_col].tolist()
                    if len(x_vals) > max_points:
                        step = max(1, len(x_vals) // max_points)
                        x_vals_ds = x_vals[::step]
                        y_vals_ds = y_vals[::step]
                    else:
                        x_vals_ds = x_vals
                        y_vals_ds = y_vals
                    area_series[y_col] = [{"x": str(xv), "y": float(yv)} for xv, yv in zip(x_vals_ds, y_vals_ds)]
                return {
                    "type": "area",
                    "series": area_series,
                    "title": title or f"Area chart of {y} by {x}",
                    "x": x,
                    "y": y,
                    "sampled": True,
                    "original_size": original_size,
                    "out_of_core": "polars"
                }
    # If not using Dask/Polars, fallback to pandas and sample as before
    if original_size > effective_max_processing_rows:
        print(f"Dataset large ({original_size:,} rows). Sampling {effective_max_processing_rows:,} rows for processing...")
        # Use systematic sampling to preserve patterns
        step = max(1, original_size // effective_max_processing_rows)
        df = df.iloc[::step].copy()
    
    if plot_type == "stacked_bar" and x and y:
        # y can be a list of columns to stack
        y_cols = y if isinstance(y, list) else [y]
        data = df[[x] + y_cols].dropna()
        grouped = data.groupby(x)[y_cols].sum().head(max_points)
        bars = {col: grouped[col].tolist() for col in y_cols}
        x_labels = grouped.index.tolist()
        return {
            "type": "stacked_bar",
            "x": x_labels,
            "bars": bars,
            "y_cols": y_cols,
            "title": title or f"Stacked Bar of {y_cols} by {x}",
            "sampled": original_size > max_processing_rows,
            "original_size": original_size
        }
    elif plot_type == "heatmap" and x and y:
        # If both x and y are numeric, do a 2D histogram (numeric heatmap)
        if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
            # 2D histogram binning
            x_vals = df[x].dropna()
            y_vals = df[y].dropna()
            # Use only overlapping indices
            valid_idx = x_vals.index.intersection(y_vals.index)
            x_vals = x_vals.loc[valid_idx]
            y_vals = y_vals.loc[valid_idx]
            # Downsample if too large
            if len(x_vals) > max_processing_rows:
                step = max(1, len(x_vals) // max_processing_rows)
                x_vals = x_vals[::step]
                y_vals = y_vals[::step]
            import numpy as np
            counts, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=bins)
            # Convert bin edges to bin centers for axes
            x_centers = 0.5 * (xedges[:-1] + xedges[1:])
            y_centers = 0.5 * (yedges[:-1] + yedges[1:])
            # Limit axes if too many bins
            if len(x_centers) > max_points:
                x_centers = x_centers[:max_points]
                counts = counts[:max_points, :]
            if len(y_centers) > max_points:
                y_centers = y_centers[:max_points]
                counts = counts[:, :max_points]
            return {
                "type": "heatmap",
                "x": [float(xc) for xc in x_centers],
                "y": [float(yc) for yc in y_centers],
                "z": counts.T.tolist(),  # transpose so z[i][j] is value at (x[i], y[j])
                "title": title or f"2D Histogram Heatmap of {x} vs {y}",
                "x_label": x,
                "y_label": y,
                "sampled": original_size > max_processing_rows,
                "original_size": original_size,
                "numeric_heatmap": True
            }
        # Otherwise, use categorical/categorical or categorical/numeric pivot as before
        if column:
            # Use the specified column as values to aggregate
            pivot = pd.pivot_table(df, index=y, columns=x, values=column, aggfunc="sum", fill_value=0)
            title_default = f"Heatmap of {column} by {y} vs {x}"
        else:
            # Fallback to counting occurrences
            pivot = pd.pivot_table(df, index=y, columns=x, aggfunc="size", fill_value=0)
            title_default = f"Heatmap count of {y} vs {x}"
        # Limit heatmap dimensions to prevent overwhelming output
        if len(pivot.columns) > max_points:
            pivot = pivot.iloc[:, :max_points]
        if len(pivot.index) > max_points:
            pivot = pivot.iloc[:max_points, :]
        return {
            "type": "heatmap",
            "x": list(pivot.columns),
            "y": list(pivot.index),
            "z": pivot.values.tolist(),
            "title": title or title_default,
            "aggregated_column": column if column else None,
            "sampled": original_size > max_processing_rows,
            "original_size": original_size
        }

    elif plot_type == "box" or plot_type == "boxplot":
        # y can be a single column or list of columns
        y_cols = y if isinstance(y, list) else [y]
        box_data = {}
        for col in y_cols:
            if col in df.columns:
                # For box plots, sample the data if too large
                col_data = df[col].dropna()
                if len(col_data) > max_processing_rows:
                    col_data = col_data.sample(n=max_processing_rows, random_state=42)
                box_data[col] = col_data.tolist()
        return {
            "type": "boxplot",
            "columns": y_cols,
            "data": box_data,
            "title": title or f"Boxplot of {y_cols}",
            "sampled": original_size > max_processing_rows,
            "original_size": original_size
        }
    elif plot_type == "histogram" and column:
        values = df[column].dropna()
        # For histogram, we can work with more data since we're binning
        if len(values) > max_processing_rows * 2:  # Allow more for histograms
            values = values.sample(n=max_processing_rows * 2, random_state=42)
        
        counts, bin_edges = np.histogram(values, bins=bins)
        return {
            "type": "histogram",
            "bins": [
                {"range": [float(bin_edges[i]), float(bin_edges[i+1])], "count": int(counts[i])}
                for i in range(len(counts))
            ],
            "title": title or f"Distribution of {column}",
            "column": column,
            "sampled": original_size > max_processing_rows * 2,
            "original_size": original_size
        }
    elif plot_type == "line":
        # Support both (x, y) and (column) usage
        if x and y:
            data = df[[x, y]].dropna()
            x_vals = data[x].tolist()
            y_vals = data[y].tolist()
            if len(x_vals) > max_points:
                step = max(1, len(x_vals) // max_points)
                x_vals = x_vals[::step]
                y_vals = y_vals[::step]
            return {
                "type": "line",
                "points": [{"x": str(xv), "y": float(yv)} for xv, yv in zip(x_vals, y_vals)],
                "title": title or f"{y} vs {x}",
                "x": x,
                "y": y,
                "sampled": original_size > max_processing_rows or len(data) > max_points,
                "original_size": original_size
            }
        elif column:
            y_data = df[column].dropna()
            x_data = y_data.index.tolist()
            if len(x_data) > max_points:
                step = max(1, len(x_data) // max_points)
                x_data = x_data[::step]
                y_data = y_data.iloc[::step]
            return {
                "type": "line",
                "points": [{"x": str(x), "y": float(y)} for x, y in zip(x_data, y_data)],
                "title": title or f"{column} Over Index",
                "column": column,
                "sampled": original_size > max_processing_rows or len(y_data) > max_points,
                "original_size": original_size
            }
        else:
            return {"error": "For line plot, provide either (column) or (x, y)"}
    elif plot_type == "bar" and x:
        if y:
            grouped = df.groupby(x)[y].sum().sort_values(ascending=False).head(max_points)
            bars = [{"label": str(idx), "value": float(val)} for idx, val in grouped.items()]
            return {
                "type": "bar", 
                "bars": bars, 
                "title": title or f"{y} by {x}", 
                "x": x, 
                "y": y,
                "sampled": original_size > max_processing_rows,
                "original_size": original_size
            }
        else:
            counts = df[x].value_counts().head(max_points)
            bars = [{"label": str(idx), "value": int(val)} for idx, val in counts.items()]
            return {
                "type": "bar", 
                "bars": bars, 
                "title": title or f"Count by {x}", 
                "x": x,
                "sampled": original_size > max_processing_rows,
                "original_size": original_size
            }

    elif plot_type == "pie":
        if x and y:
            grouped = df.groupby(x)[y].sum().sort_values(ascending=False).head(max_points)
            slices = [{"label": str(idx), "value": float(val)} for idx, val in grouped.items()]
            return {
                "type": "pie", 
                "slices": slices, 
                "title": title or f"{y} by {x}", 
                "x": x, 
                "y": y,
                "sampled": original_size > max_processing_rows,
                "original_size": original_size
            }
        elif column or x:
            col = column or x
            value_counts = df[col].value_counts().head(max_points)
            slices = [{"label": str(idx), "value": int(val)} for idx, val in value_counts.items()]
            return {
                "type": "pie", 
                "slices": slices, 
                "title": title or f"Pie chart of {col}", 
                "column": col,
                "sampled": original_size > max_processing_rows,
                "original_size": original_size
            }
    elif plot_type == "area":
        # Area chart: robust, scalable split-by/category support
        if x and y:
            import re
            max_categories = 20
            auto_limit = True
            # Try to get from args if passed (for backward compatibility)
            import inspect
            frame = inspect.currentframe()
            outer = inspect.getouterframes(frame)
            for f in outer:
                if 'args' in f.frame.f_locals:
                    args = f.frame.f_locals['args']
                    if hasattr(args, 'max_categories'):
                        max_categories = getattr(args, 'max_categories', 20)
                    if hasattr(args, 'auto_limit'):
                        auto_limit = getattr(args, 'auto_limit', True)
                    break
            # 1. If column is provided, require it to be valid, else error (do not infer)
            if column is not None:
                if column not in df.columns:
                    return {
                        "error": f"Provided split-by column '{column}' not found in DataFrame columns.",
                        "x": x,
                        "y": y,
                        "original_size": original_size
                    }
                split_column = column
            else:
                split_column = None
                # 2. Try to infer from title (e.g., "split by ..." or "per ...")
                if title:
                    m = re.search(r"split by ([\w_]+)", title, re.IGNORECASE)
                    if not m:
                        m = re.search(r"per ([\w_]+)", title, re.IGNORECASE)
                    if m:
                        candidate = m.group(1)
                        for col in df.columns:
                            if col == candidate or col.lower() == candidate.lower() or candidate.lower() in col.lower():
                                split_column = col
                                break
                # 3. Data-driven: pick a categorical/object column (not x/y) with few unique values
                if not split_column:
                    sample_size = min(5000, len(df))
                    df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
                    candidates = [col for col in df.columns if col not in [x, y] and (df[col].dtype == object or str(df[col].dtype).startswith("category"))]
                    best = None
                    best_card = None
                    for col in candidates:
                        nunique = df_sample[col].nunique(dropna=True)
                        if 2 <= nunique <= max_categories:
                            if best is None or nunique < best_card:
                                best = col
                                best_card = nunique
                    if best:
                        split_column = best
            # If a valid split_column is found, check its cardinality on the full df (with a hard cap)
            if split_column and split_column in df.columns:
                cardinality = df[split_column].nunique(dropna=True)
                if cardinality > max_categories:
                    if auto_limit:
                        # Fallback: use top-N categories
                        top_n = max_categories
                        top_categories = df.groupby(split_column)[y].sum().nlargest(top_n).index
                        df_filtered = df[df[split_column].isin(top_categories)]
                        groupby_sample_size = min(10000, len(df_filtered))
                        df_group = df_filtered.sample(n=groupby_sample_size, random_state=42) if len(df_filtered) > groupby_sample_size else df_filtered
                        grouped = df_group[[x, split_column, y]].dropna().groupby([x, split_column])[y].sum().reset_index()
                        area_series = {}
                        for series_name in grouped[split_column].unique():
                            series_data = grouped[grouped[split_column] == series_name]
                            x_vals = series_data[x].tolist()
                            y_vals = series_data[y].tolist()
                            if len(x_vals) > max_points:
                                step = max(1, len(x_vals) // max_points)
                                x_vals = x_vals[::step]
                                y_vals = y_vals[::step]
                            area_series[str(series_name)] = [
                                {"x": str(xv), "y": float(yv)} for xv, yv in zip(x_vals, y_vals)
                            ]
                        return {
                            "type": "area",
                            "series": area_series,
                            "title": f"{title or f'Area chart of {y} by {x}'} (Top {top_n})",
                            "x": x,
                            "y": y,
                            "series_column": split_column,
                            "truncated": True,
                            "total_categories": cardinality,
                            "showing_categories": top_n,
                            "sampled": True if len(df_filtered) > groupby_sample_size else False,
                            "original_size": original_size
                        }
                    else:
                        # Return error and alternatives
                        return {
                            "error": f"Split-by column '{split_column}' has too many unique values ({cardinality}). Please specify a column with fewer categories for area chart.",
                            "x": x,
                            "y": y,
                            "original_size": original_size,
                            "alternatives": [
                                {
                                    "plot_type": "area",
                                    "description": f"Area chart showing top {max_categories} {split_column}s only",
                                    "params": {
                                        "df_name": None,  # frontend should fill
                                        "plot_type": "area",
                                        "x": x,
                                        "y": y,
                                        "column": split_column,
                                        "title": f"{title} (Top {max_categories})",
                                        "auto_limit": True,
                                        "max_categories": max_categories
                                    }
                                },
                                {
                                    "plot_type": "line",
                                    "description": "Aggregated trend without product split",
                                    "params": {
                                        "df_name": None,
                                        "plot_type": "line",
                                        "x": x,
                                        "y": y,
                                        "title": f"Total {y} Over Time"
                                    }
                                },
                                {
                                    "plot_type": "bar",
                                    "description": f"Top {max_categories} {split_column}s by total {y}",
                                    "params": {
                                        "df_name": None,
                                        "plot_type": "bar",
                                        "x": split_column,
                                        "y": y,
                                        "title": f"Top {split_column}s by {y}"
                                    }
                                }
                            ]
                        }
                # For large DataFrames, sample for groupby as well
                groupby_sample_size = min(10000, len(df))
                df_group = df.sample(n=groupby_sample_size, random_state=42) if len(df) > groupby_sample_size else df
                grouped = df_group[[x, split_column, y]].dropna().groupby([x, split_column])[y].sum().reset_index()
                area_series = {}
                for series_name in grouped[split_column].unique():
                    series_data = grouped[grouped[split_column] == series_name]
                    x_vals = series_data[x].tolist()
                    y_vals = series_data[y].tolist()
                    if len(x_vals) > max_points:
                        step = max(1, len(x_vals) // max_points)
                        x_vals = x_vals[::step]
                        y_vals = y_vals[::step]
                    area_series[str(series_name)] = [
                        {"x": str(xv), "y": float(yv)} for xv, yv in zip(x_vals, y_vals)
                    ]
                return {
                    "type": "area",
                    "series": area_series,
                    "title": title or f"Area chart of {y} by {x} per {split_column}",
                    "x": x,
                    "y": y,
                    "series_column": split_column,
                    "sampled": True if len(df) > groupby_sample_size else False,
                    "original_size": original_size
                }
            else:
                # Fallback: treat y as a list of columns (legacy behavior)
                y_cols = y if isinstance(y, list) else [y]
                data = df[[x] + y_cols].dropna()
                x_vals = data[x].tolist()
                area_series = {}
                for y_col in y_cols:
                    y_vals = data[y_col].tolist()
                    if len(x_vals) > max_points:
                        step = max(1, len(x_vals) // max_points)
                        x_vals_ds = x_vals[::step]
                        y_vals_ds = y_vals[::step]
                    else:
                        x_vals_ds = x_vals
                        y_vals_ds = y_vals
                    area_series[y_col] = [{"x": str(xv), "y": float(yv)} for xv, yv in zip(x_vals_ds, y_vals_ds)]
                return {
                    "type": "area",
                    "series": area_series,
                    "title": title or f"Area chart of {y} by {x}",
                    "x": x,
                    "y": y,
                    "sampled": original_size > max_processing_rows,
                    "original_size": original_size
                }
        else:
            return {"error": "For area chart, provide x and y (y can be a list), and optionally column for series"}
    elif plot_type == "scatter" and x and y:
        data = df[[x, y]].dropna()
        x_vals = data[x].tolist()
        y_vals = data[y].tolist()
        if len(x_vals) > max_points:
            step = max(1, len(x_vals) // max_points)
            x_vals = x_vals[::step]
            y_vals = y_vals[::step]
        return {
            "type": "scatter",
            "points": [{"x": str(xv), "y": float(yv)} for xv, yv in zip(x_vals, y_vals)],
            "title": title or f"Scatterplot of {y} vs {x}",
            "x": x,
            "y": y,
            "sampled": original_size > max_processing_rows or len(data) > max_points,
            "original_size": original_size
        }
    else:
        return {"error": "Supported: histogram (column), line (column or x/y), bar (x, y), stacked_bar (x, y[list]), pie (column), area (x, y), scatter (x, y), heatmap (x, y), boxplot (y)"}

@mcp.tool()
def create_visualization(args: CreateVisualizationArgs) -> list:
    """Create visualization data for React frontend.
    This tool extracts data for various chart types from a DataFrame.
    Do list all supported chart types using list_supported_chart_types tool.
    Do list_dataframes tool before using this tool to ensure you know the available DataFrames and their columns.
    """
    global _dataframes
    df_name = args.df_name
    if df_name not in _dataframes:
        return [TextContent(type="text", text=f"DataFrame '{df_name}' not found. Available: {list(_dataframes.keys())}")]
    df = _dataframes[df_name]
    plot_type = args.plot_type.lower() if isinstance(args.plot_type, str) else args.plot_type
    # Validate and parse x/y for each chart type
    x = args.x
    y = args.y
    # Only allow y as list for certain chart types
    y_list_types = ["stacked_bar", "area"]
    x_list_types = []  # No chart type currently supports x as a list
    if plot_type in y_list_types:
        if isinstance(y, str) and "," in y:
            y = [col.strip() for col in y.split(",")]
    else:
        # For all other chart types, y must be a string (single column)
        if isinstance(y, list):
            return [TextContent(type="text", text=f"Error: Chart type '{plot_type}' does not support multiple y columns.")]
        if isinstance(y, str) and "," in y:
            y_split = [col.strip() for col in y.split(",")]
            if len(y_split) > 1:
                return [TextContent(type="text", text=f"Error: Chart type '{plot_type}' does not support multiple y columns.")]
            y = y_split[0]
    if plot_type in x_list_types:
        if isinstance(x, str) and "," in x:
            x = [col.strip() for col in x.split(",")]
    else:
        if isinstance(x, list):
            return [TextContent(type="text", text=f"Error: Chart type '{plot_type}' does not support multiple x columns.")]
        if isinstance(x, str) and "," in x:
            x_split = [col.strip() for col in x.split(",")]
            if len(x_split) > 1:
                return [TextContent(type="text", text=f"Error: Chart type '{plot_type}' does not support multiple x columns.")]
            x = x_split[0]
    plot_data = _extract_plot_data(
        df,
        plot_type,
        x=x,
        y=y,
        column=args.column,
        title=args.title,
        bins=args.bins,
        max_points=args.max_points
    )
    return [TextContent(type="text", text=json.dumps(plot_data, indent=2))]


# @mcp.tool()
# def create_visualization(args: CreateVisualizationArgs) -> list:
#     """Create visualization data for React frontend.
#     This tool extracts data for various chart types from a DataFrame.
#     Do list all supported chart types using list_supported_chart_types tool.
#     Do list_dataframes tool before using this tool to ensure you know the available DataFrames and their columns.
#     """
#     global _dataframes
#     df_name = args.df_name
#     if df_name not in _dataframes:
#         return [TextContent(type="text", text=f"DataFrame '{df_name}' not found. Available: {list(_dataframes.keys())}")]
#     df = _dataframes[df_name]
#     # Accept x and y as comma-separated string or list
#     x = args.x
#     y = args.y
#     if isinstance(x, str) and "," in x:
#         x = [col.strip() for col in x.split(",")]
#     if isinstance(y, str) and "," in y:
#         y = [col.strip() for col in y.split(",")]
#     plot_type = args.plot_type.lower() if isinstance(args.plot_type, str) else args.plot_type

#     plot_data = _extract_plot_data(
#         df,
#         plot_type,
#         x=x,
#         y=y,
#         column=args.column,
#         title=args.title,
#         bins=args.bins,
#         max_points=args.max_points
#     )
#     return [TextContent(type="text", text=json.dumps(plot_data, indent=2))]
# Tool: List supported chart types
@mcp.tool()
def list_supported_chart_types() -> list:
    """List all supported chart types for visualization."""
    chart_types = [
        "histogram",
        "line",
        "bar",
        "stacked_bar",
        "pie",
        "area",
        "scatter",
        "heatmap",
        "boxplot"
    ]
    return [TextContent(type="text", text="Supported chart types: " + ", ".join(chart_types))]

class PreviewDataFrameArgs(BaseModel):
    df_name: str
    n: int = 5  # Number of rows to preview

# @mcp.tool()
# def preview_dataframe(args: PreviewDataFrameArgs) -> list:
#     """Preview the first n rows of a DataFrame in memory."""
#     global _dataframes
#     df_name = args.df_name
#     n = args.n
#     if df_name not in _dataframes:
#         return [TextContent(type="text", text=f"DataFrame '{df_name}' not found. Available: {list(_dataframes.keys())}")]
#     df = _dataframes[df_name]
#     preview = df.head(n).to_string(index=False)
#     return [TextContent(type="text", text=f"Preview of '{df_name}' (first {n} rows):\n{preview}")]



# --- Table preview resource pattern ---
_last_table_preview = None  # Global cache for last prepared table preview


class PrepareTableResourceArgs(BaseModel):
    df_name: str
    n: int = 5
    title: str = None

@mcp.tool()
def prepare_table_resource(args: PrepareTableResourceArgs) -> list:
    """Prepare a DataFrame preview for the frontend resource in CSV format. Does NOT return the table to the LLM."""
    global _dataframes, _last_table_preview
    df_name = args.df_name
    n = args.n
    title = args.title if args.title else f"{df_name} (first {n} rows)"
    if df_name not in _dataframes:
        return [TextContent(type="text", text=f"DataFrame '{df_name}' not found. Available: {list(_dataframes.keys())}")]
    df = _dataframes[df_name]
    csv_data = df.head(n).to_csv(index=False)
    _last_table_preview = None  # Reset previous preview
    _last_table_preview = {
        "title": title,
        "csv": csv_data
    }
    return [TextContent(type="text", text=f"Table preview for '{df_name}' ({n} rows, CSV format) prepared. Fetch via resource.")]

@mcp.resource("data-exploration://table-preview", mime_type="application/json")
def table_preview_resource():
    """Return the last prepared table preview (for frontend display only)."""
    global _last_table_preview
    if _last_table_preview is None:
        return {"error": "No table preview prepared. Use the prepare_table_resource tool first."}
    return _last_table_preview
# @mcp.tool()
# def preview_dataframe(args: PreviewDataFrameArgs) -> list:
#     """Preview the first n rows of a DataFrame in memory, but do NOT return the table to the LLM."""
#     global _dataframes
#     df_name = args.df_name
#     n = args.n
#     if df_name not in _dataframes:
#         return [TextContent(type="text", text=f"DataFrame '{df_name}' not found. Available: {list(_dataframes.keys())}")]
#     df = _dataframes[df_name]
#     # Print preview to server log for developer/debugging
#     print(f"Preview of '{df_name}' (first {n} rows):\n{df.head(n).to_string(index=False)}")
#     # Return only summary to LLM
#     summary = f"DataFrame '{df_name}': shape={df.shape}, columns={list(df.columns)}, dtypes={dict(df.dtypes)}"
#     return [TextContent(type="text", text=summary + "\n[Preview printed to server log, not included in LLM response.]")]

### Data Exploration Tools Description & Schema
_dataframes: Dict[str, pd.DataFrame] = {}

### Prompt templates
class DataExplorationPrompts(str, Enum):
    EXPLORE_DATA = "explore-data"

class PromptArgs(str, Enum):
    CSV_PATH = "csv_path"
    TOPIC = "topic"

# PROMPT_TEMPLATE = """
# You are a professional Data Scientist tasked with performing exploratory data analysis on a dataset. Your goal is to provide insightful analysis while ensuring stability and manageable result sizes.

# First, load the CSV file from the following path:

# <csv_path>
# {csv_path}
# </csv_path>

# Your analysis should focus on the following topic:

# <analysis_topic>
# {topic}
# </analysis_topic>

# You have access to the following tools for your analysis:
# 1. load_csv: Use this to load the CSV file.
# 2. list_dataframes: Use this to see all available DataFrames and their columns.
# 3. list_supported_chart_types: Use this to see all supported chart types for visualization.
# 4. run_script: Use this to execute Python scripts on the MCP server.
# 5. create_visualization: Use this to create charts from DataFrames.

# Important: Before using run_script or create_visualization, always call list_dataframes to discover available DataFrames and their columns, and call list_supported_chart_types to discover supported chart types.

# Please follow these steps carefully:

# 1. Load the CSV file using the load_csv tool.

# 2. Call list_dataframes to see all available DataFrames and their columns.

# 3. Call list_supported_chart_types to see all supported chart types for visualization.

# 4. Explore the dataset. Provide a brief summary of its structure, including the number of rows, columns, and data types. Wrap your exploration process in <dataset_exploration> tags, including:
#    - List of key statistics about the dataset
#    - Potential challenges you foresee in analyzing this data

# 5. Wrap your thought process in <analysis_planning> tags:
#    Analyze the dataset size and complexity:
#    - How many rows and columns does it have?
#    - Are there any potential computational challenges based on the data types or volume?
#    - What kind of questions would be appropriate given the dataset's characteristics and the analysis topic?
#    - How can we ensure that our questions won't result in excessively large outputs?

#    Based on this analysis:
#    - List 10 potential questions related to the analysis topic
#    - Evaluate each question against the following criteria:
#      * Directly related to the analysis topic
#      * Can be answered with reasonable computational effort
#      * Will produce manageable result sizes
#      * Provides meaningful insights into the data
#    - Select the top 5 questions that best meet all criteria

# 6. List the 5 questions you've selected, ensuring they meet the criteria outlined above.

# 7. For each question, follow these steps:
#    a. Wrap your thought process in <analysis_planning> tags:
#       - How can I structure the Python script to efficiently answer this question?
#       - What data preprocessing steps are necessary?
#       - How can I limit the output size to ensure stability?
#       - What type of visualization would best represent the results?
#       - Outline the main steps the script will follow
   
#    b. Write a Python script to answer the question. Include comments explaining your approach and any measures taken to limit output size.
   
#    c. Use the run_script tool to execute your Python script on the MCP server.
   

# 8. After completing the analysis for all 5 questions, provide a brief summary of your findings and any overarching insights gained from the data.

# Remember to prioritize stability and manageability in your analysis. If at any point you encounter potential issues with large result sets, adjust your approach accordingly.

# Please begin your analysis by loading the CSV file and providing an initial exploration of the dataset.
# """
PROMPT_TEMPLATE = """
You are a professional Data Scientist. Perform exploratory data analysis on the dataset at this path:

<csv_path>
{csv_path}
</csv_path>

Focus your analysis on: **{topic}**

## Your Task:
1. Load and explore the dataset structure
2. Identify 3-5 key insights related to {topic}
3. Create appropriate visualizations to support your findings
4. Provide a concise summary of your analysis

## Important Guidelines:
- Always call `list_dataframes` before using `run_script` or `create_visualization`
- Use `list_supported_chart_types` to discover available chart types before `create_visualization`
- Focus on the most relevant insights for the given topic
- Use visualizations to enhance understanding

Begin by loading the CSV file and exploring its basic structure.
"""
# - Keep individual script outputs manageable (limit large results)


# PROMPT_TEMPLATE = """
# You are a professional Data Scientist. Perform exploratory data analysis on the dataset at this path:

# <csv_path>
# {csv_path}
# </csv_path>

# Focus your analysis on: **{topic}**

# ## Your Task (Stepwise):
# 1. Load the dataset using `load_csv`.
# 2. Preview the first 5 rows using `preview_dataframe`.
# 3. List all DataFrames in memory using `list_dataframes`.

# **STOP after these steps and wait for further instructions.**

# Once confirmed, proceed to:
# 4. Run analysis scripts with `run_script`.
# 5. Create visualizations with `create_visualization`.

# **Always keep each request to 3–4 tool calls maximum to avoid step-limit errors.**
# """

#    d. Render the results returned by the run_script tool as a chart using plotly.js (prefer loading from cdnjs.cloudflare.com). Do not use react or recharts, and do not read the original CSV file directly. Provide the plotly.js code to generate the chart.


class DataExplorationTools(str, Enum):



    LOAD_CSV = "load_csv"
    RUN_SCRIPT = "run_script"

LOAD_CSV_TOOL_DESCRIPTION = """
Load CSV File Tool

Purpose:
Load a local CSV file into a DataFrame.

Usage Notes:
    •	If a df_name is not provided, the tool will automatically assign names sequentially as df_1, df_2, and so on.
    •	NEVER load CSV files in run_script - use this tool exclusively for CSV loading.
"""

RUN_SCRIPT_TOOL_DESCRIPTION = """
Python Script Execution Tool

Purpose:
Execute Python scripts for specific data analytics tasks.

Allowed Actions
    1.	Print Results: Output will be displayed as the script's stdout.
    2.	[Optional] Save DataFrames: Store DataFrames in memory for future use by specifying a save_to_memory name.
    3.	Data Analysis: Calculate statistics, aggregations, correlations, etc.
    4.	Data Processing: Clean, transform, filter, group data.
    5.	DataFrame Charts: Use DataFrame.plot() methods - charts will auto-display in frontend!

Prohibited Actions
    1.	Overwriting Original DataFrames: Do not modify existing DataFrames to preserve their integrity for future tasks.
    2.	Loading CSV files: Use load_csv tool instead. Scripts should only work with DataFrames already in memory.
    3.	Direct matplotlib usage: Use DataFrame.plot() methods instead of plt.show(), plt.figure(), etc.
    4.	Saving Files: Scripts should not save files, only print results and create charts.

Available DataFrames: df_1, df_2, etc. (loaded via load_csv tool)

Chart Examples (Auto-Display):
- df.plot(kind='bar', title='Bar Chart') - Bar chart
- df.plot(kind='line', x='date', y='value') - Line chart
- df['column'].value_counts().plot(kind='pie') - Pie chart
- df.plot(kind='scatter', x='col1', y='col2') - Scatter plot
- df.plot(kind='hist', bins=20) - Histogram

Best Practices:
- Use df.plot() methods for charts that will auto-display
- Print summary statistics and insights
- Combine charts with printed analysis for complete insights
- Focus on meaningful visualizations that tell a story
"""

@mcp.prompt()
def explore_data_prompt(csv_path: str, topic: str = "general data exploration"):
    """A prompt to explore a CSV dataset as a data scientist"""
    print("Registering explore_data_prompt")
    return PROMPT_TEMPLATE.format(csv_path=csv_path, topic=topic)


# Resource: DataFrame schema and preview
@mcp.resource("data-exploration://schema", mime_type="application/json")
def schema_resource():
    """All DataFrames in memory with columns, dtypes, shape, and preview rows."""
    print("[RESOURCE] schema_resource called")
    global _dataframes
    schema = {}
    for name, df in _dataframes.items():
        schema[name] = {
            "columns": list(df.columns),
            "dtypes": {col: str(df.dtypes[col]) for col in df.columns},
            "shape": list(df.shape),
            "preview": df.head(5).to_dict(orient="records")
        }
    return schema

# Keep notes resource as well
@mcp.resource("data-exploration://notes", mime_type="text/plain")
def notes_resource():
    """Notes generated by the data exploration server."""
    print("[RESOURCE] notes_resource called")
    global _notes
    return "\n".join(_notes) if _notes else "No notes yet. Use load_csv or run_script to generate notes."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
