
from enum import Enum
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, Prompt, PromptArgument, Resource  
import pandas as pd
import numpy as np
import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

mcp = FastMCP(name="dsdb", host="127.0.0.1", port=8004)

# In-memory data store for loaded DataFrames
_dataframes = {}
_df_count = 0
_notes: list[str] = []
_exploration_history: list[Dict[str, Any]] = []  # Track exploration steps
_data_insights: Dict[str, Any] = {}  # Store discovered insights
_hypotheses: list[Dict[str, Any]] = []  # Track hypotheses and validations

# Global DB engine and session
db_engine = None
db_session = None

def _next_df_name():
    global _df_count
    _df_count += 1
    return f"df_{_df_count}"

def _log_exploration_step(step_type: str, operation: str, result: str, insights: Dict[str, Any] = None):
    """Log each exploration step for analysis"""
    global _exploration_history
    step = {
        "step_type": step_type,
        "operation": operation,
        "result": result,
        "insights": insights or {},
        "timestamp": pd.Timestamp.now().isoformat()
    }
    _exploration_history.append(step)



# --- DB Connection Args ---
class LoadCsvArgs(BaseModel):
    csv_path: str
    df_name: Optional[str] = None


# Remove LoadDbArgs and load_db tool, user will handle DB connection manually


class RunScriptArgs(BaseModel):
    script: str
    save_to_memory: Optional[List[str]] = None

class EnhancedSqlQueryArgs(BaseModel):
    sql: str
    df_name: Optional[str] = None
    purpose: Optional[str] = None  # Track why this query is being run
    hypothesis: Optional[str] = None  # What hypothesis is being tested

@mcp.tool()
def run_sql_query_enhanced(args: EnhancedSqlQueryArgs) -> list:
    """Enhanced SQL query with exploration tracking and automatic insights generation."""
    global _dataframes, _notes, _data_insights
    global db_engine  # <-- Add this line

    sql = args.sql
    df_name = args.df_name or _next_df_name()
    purpose = args.purpose or "General query"
    hypothesis = args.hypothesis
    try:
        if db_engine is None:
            return [TextContent(type="text", text="Database engine is not initialized.")]
        from sqlalchemy import text
        with db_engine.connect() as conn:
            df = pd.read_sql_query(text(sql), conn)
        _dataframes[df_name] = df
        # Generate automatic insights
        insights = _generate_dataframe_insights(df, df_name)
        _data_insights[df_name] = insights
        # Log exploration step
        _log_exploration_step(
            step_type="sql_query",
            operation=sql,
            result=f"Loaded {len(df)} rows into {df_name}",
            insights=insights
        )
        # Validate hypothesis if provided
        if hypothesis:
            validation = _validate_hypothesis(hypothesis, df, insights)
            _hypotheses.append({
                "hypothesis": hypothesis,
                "validation": validation,
                "dataframe": df_name,
                "timestamp": pd.Timestamp.now().isoformat()
            })
        result_text = f"Loaded SQL query into dataframe '{df_name}' ({len(df)} rows)\n"
        result_text += f"Purpose: {purpose}\n"
        result_text += f"Automatic insights: {insights.get('summary', 'No insights generated')}"
        _notes.append(result_text)
        return [TextContent(type="text", text=result_text)]
    except Exception as e:
        error_msg = f"Error running SQL: {str(e)}"
        _log_exploration_step("sql_error", sql, error_msg)
        _notes.append(error_msg)
        return [TextContent(type="text", text=error_msg)]
    

    
def _generate_dataframe_insights(df: pd.DataFrame, df_name: str) -> Dict[str, Any]:
    """Generate automatic insights about a DataFrame (inspired by RAISE's data understanding)"""
    insights = {}
    try:
        # Basic statistics
        insights["shape"] = df.shape
        insights["columns"] = list(df.columns)
        insights["dtypes"] = df.dtypes.to_dict()
        # Data quality insights
        insights["missing_values"] = df.isnull().sum().to_dict()
        insights["duplicate_rows"] = df.duplicated().sum()
        # Numeric columns insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights["numeric_summary"] = df[numeric_cols].describe().to_dict()
        # Categorical insights
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            insights["categorical_summary"] = {}
            for col in categorical_cols:
                insights["categorical_summary"][col] = {
                    "unique_values": df[col].nunique(),
                    "top_values": df[col].value_counts().head(5).to_dict()
                }
        # Potential issues detection
        issues = []
        if df.duplicated().sum() > 0:
            issues.append(f"Found {df.duplicated().sum()} duplicate rows")
        high_missing = df.isnull().sum()
        high_missing_cols = high_missing[high_missing > len(df) * 0.1].index.tolist()
        if high_missing_cols:
            issues.append(f"High missing values in columns: {high_missing_cols}")
        insights["potential_issues"] = issues
        # Generate summary
        insights["summary"] = f"DataFrame with {df.shape[0]} rows and {df.shape[1]} columns. "
        if issues:
            insights["summary"] += f"Potential issues: {'; '.join(issues)}"
        else:
            insights["summary"] += "No obvious data quality issues detected."
    except Exception as e:
        insights["error"] = f"Error generating insights: {str(e)}"
    return insights

def _validate_hypothesis(hypothesis: str, df: pd.DataFrame, insights: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a hypothesis against the data"""
    validation = {
        "hypothesis": hypothesis,
        "status": "unknown",
        "evidence": [],
        "confidence": 0.0
    }
    try:
        # Simple keyword-based validation (could be enhanced with NLP)
        if "missing" in hypothesis.lower() or "null" in hypothesis.lower():
            missing_info = insights.get("missing_values", {})
            if any(count > 0 for count in missing_info.values()):
                validation["status"] = "supported"
                validation["evidence"].append(f"Missing values found: {missing_info}")
                validation["confidence"] = 0.8
            else:
                validation["status"] = "not_supported"
                validation["evidence"].append("No missing values found")
                validation["confidence"] = 0.9
        elif "duplicate" in hypothesis.lower():
            dup_count = insights.get("duplicate_rows", 0)
            if dup_count > 0:
                validation["status"] = "supported"
                validation["evidence"].append(f"Found {dup_count} duplicate rows")
                validation["confidence"] = 0.9
            else:
                validation["status"] = "not_supported"
                validation["evidence"].append("No duplicate rows found")
                validation["confidence"] = 0.9
        # Add more hypothesis validation patterns as needed
    except Exception as e:
        validation["error"] = f"Error validating hypothesis: {str(e)}"
    return validation

class DataExplorationArgs(BaseModel):
    df_name: str
    exploration_type: str  # "quality", "relationships", "patterns", "anomalies"
    specific_columns: Optional[List[str]] = None

@mcp.tool()
def explore_data_patterns(args: DataExplorationArgs) -> list:
    """Systematic data exploration inspired by RAISE's database exploration strategy"""
    global _dataframes, _data_insights
    df_name = args.df_name
    exploration_type = args.exploration_type
    specific_columns = args.specific_columns
    if df_name not in _dataframes:
        return [TextContent(type="text", text=f"DataFrame '{df_name}' not found. Available: {list(_dataframes.keys())}")]
    df = _dataframes[df_name]
    results = []
    try:
        if exploration_type == "quality":
            quality_report = _explore_data_quality(df, specific_columns)
            results.append(f"Data Quality Report for {df_name}:")
            results.append(quality_report)
        elif exploration_type == "relationships":
            relationship_report = _explore_relationships(df, specific_columns)
            results.append(f"Relationship Analysis for {df_name}:")
            results.append(relationship_report)
        elif exploration_type == "patterns":
            pattern_report = _explore_patterns(df, specific_columns)
            results.append(f"Pattern Analysis for {df_name}:")
            results.append(pattern_report)
        elif exploration_type == "anomalies":
            anomaly_report = _detect_anomalies(df, specific_columns)
            results.append(f"Anomaly Detection for {df_name}:")
            results.append(anomaly_report)
        result_text = "\n".join(results)
        _log_exploration_step(
            step_type="data_exploration",
            operation=f"{exploration_type} exploration on {df_name}",
            result=result_text[:200] + "..." if len(result_text) > 200 else result_text
        )
        return [TextContent(type="text", text=result_text)]
    except Exception as e:
        error_msg = f"Error in data exploration: {str(e)}"
        return [TextContent(type="text", text=error_msg)]

def _explore_data_quality(df: pd.DataFrame, columns: Optional[List[str]] = None) -> str:
    if columns:
        df = df[columns]
    report = []
    missing = df.isnull().sum()
    if missing.sum() > 0:
        report.append("Missing Values:")
        for col, count in missing.items():
            if count > 0:
                pct = (count / len(df)) * 100
                report.append(f"  {col}: {count} ({pct:.1f}%)")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        report.append(f"Duplicate rows: {duplicates}")
    report.append("Data Types:")
    for col, dtype in df.dtypes.items():
        report.append(f"  {col}: {dtype}")
    return "\n".join(report) if report else "No data quality issues detected"

def _explore_relationships(df: pd.DataFrame, columns: Optional[List[str]] = None) -> str:
    if columns:
        df = df[columns]
    report = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr.append(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_val:.3f}")
        if high_corr:
            report.append("High Correlations (>0.7):")
            report.extend([f"  {item}" for item in high_corr])
    return "\n".join(report) if report else "No significant relationships detected"

def _explore_patterns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> str:
    if columns:
        df = df[columns]
    report = []
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.1:
            top_values = df[col].value_counts().head(3)
            report.append(f"{col} - Top values: {top_values.to_dict()}")
    return "\n".join(report) if report else "No significant patterns detected"

def _detect_anomalies(df: pd.DataFrame, columns: Optional[List[str]] = None) -> str:
    if columns:
        df = df[columns]
    report = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            report.append(f"{col}: {len(outliers)} outliers detected")
    return "\n".join(report) if report else "No anomalies detected"

class HypothesisArgs(BaseModel):
    hypothesis: str
    df_name: str
    test_query: Optional[str] = None

@mcp.tool()
def test_hypothesis(args: HypothesisArgs) -> list:
    """Test a hypothesis about the data"""
    global _dataframes, _hypotheses
    hypothesis = args.hypothesis
    df_name = args.df_name
    test_query = args.test_query
    if df_name not in _dataframes:
        return [TextContent(type="text", text=f"DataFrame '{df_name}' not found")]
    df = _dataframes[df_name]
    if test_query:
        try:
            result = eval(test_query, {"df": df, "pd": pd, "np": np})
            validation = {
                "hypothesis": hypothesis,
                "test_query": test_query,
                "result": str(result),
                "status": "tested",
                "timestamp": pd.Timestamp.now().isoformat()
            }
            _hypotheses.append(validation)
            return [TextContent(type="text", text=f"Hypothesis: {hypothesis}\nTest: {test_query}\nResult: {result}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error testing hypothesis: {str(e)}")]
    insights = _data_insights.get(df_name, {})
    validation = _validate_hypothesis(hypothesis, df, insights)
    _hypotheses.append(validation)
    result_text = f"Hypothesis: {hypothesis}\n"
    result_text += f"Status: {validation['status']}\n"
    result_text += f"Evidence: {'; '.join(validation['evidence'])}\n"
    result_text += f"Confidence: {validation['confidence']}"
    return [TextContent(type="text", text=result_text)]

@mcp.tool()
def get_exploration_history() -> list:
    """Get the complete exploration history with insights"""
    global _exploration_history, _hypotheses
    if not _exploration_history:
        return [TextContent(type="text", text="No exploration history available")]
    result = "=== EXPLORATION HISTORY ===\n\n"
    for i, step in enumerate(_exploration_history, 1):
        result += f"Step {i}: {step['step_type']}\n"
        result += f"  Operation: {step['operation']}\n"
        result += f"  Result: {step['result']}\n"
        if step['insights']:
            result += f"  Insights: {step['insights']}\n"
        result += f"  Time: {step['timestamp']}\n\n"
    if _hypotheses:
        result += "=== HYPOTHESIS VALIDATION HISTORY ===\n\n"
        for i, hyp in enumerate(_hypotheses, 1):
            result += f"Hypothesis {i}: {hyp['hypothesis']}\n"
            result += f"  Status: {hyp['status']}\n"
            if 'evidence' in hyp:
                result += f"  Evidence: {'; '.join(hyp['evidence'])}\n"
            result += f"  Time: {hyp['timestamp']}\n\n"
    return [TextContent(type="text", text=result)]
# Tool: List all MySQL table names and their columns/attributes
class ListTablesArgs(BaseModel):
    pass

@mcp.tool()
def list_db_tables(args: ListTablesArgs = None) -> list:
    """List all table names in the connected MySQL database and their columns/attributes.
    Uses environment variables MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DB.
    """
    global db_engine
    from sqlalchemy import inspect
    try:
        if db_engine is None:
            return [TextContent(type="text", text="Database engine is not initialized. Startup connection failed or not configured.")]
        inspector = inspect(db_engine)
        tables = inspector.get_table_names()
        result = "=== DATABASE TABLES AND COLUMNS ===\n\n"
        for table in tables:
            columns = inspector.get_columns(table)
            col_str = ", ".join([f"{col['name']} ({col['type']})" for col in columns])
            result += f"Table: {table}\n  Columns: {col_str}\n\n"
        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error listing tables: {str(e)}")]


# @mcp.tool()
# def load_csv(args: LoadCsvArgs) -> list:
#     """Load a local CSV file into a DataFrame."""
#     global _dataframes, _notes
#     csv_path = args.csv_path
#     df_name = args.df_name
#     if not df_name:
#         df_name = _next_df_name()
#     try:
#         _dataframes[df_name] = pd.read_csv(csv_path, encoding='latin1')
#         _notes.append(f"Successfully loaded CSV into dataframe '{df_name}'")
#         return [TextContent(type="text", text=f"Successfully loaded CSV into dataframe '{df_name}'")]
#     except Exception as e:
#         error_msg = f"Error loading CSV: {str(e)}"
#         _notes.append(error_msg)
#         return [TextContent(type="text", text=error_msg)]


# (DB connection and loading should be done manually by the user in a script, not as a tool)
    
@mcp.tool()
def get_notes() -> list:
    """Return the notes generated by the data exploration server."""
    global _notes
    return [TextContent(type="text", text="\n".join(_notes))]

class PreviewDataFrameArgs(BaseModel):
    df_name: str
    n: int = 5  # Number of rows to preview

@mcp.tool()
def preview_dataframe(args: PreviewDataFrameArgs) -> list:
    """Preview the first n rows of a DataFrame in memory."""
    global _dataframes
    df_name = args.df_name
    n = args.n
    if df_name not in _dataframes:
        return [TextContent(type="text", text=f"DataFrame '{df_name}' not found. Available: {list(_dataframes.keys())}")]
    df = _dataframes[df_name]
    preview = df.head(n).to_string(index=False)
    return [TextContent(type="text", text=f"Preview of '{df_name}' (first {n} rows):\n{preview}")]


@mcp.tool()
def run_script(args: RunScriptArgs) -> list:
    """Execute a Python script with enhanced tracking and insights"""
    global _dataframes, _notes
    # Your existing run_script implementation, but add exploration tracking
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
    import pandas as pd
    import numpy as np
    import sys
    import io
    import traceback
    local_vars = {**_dataframes, **memory}
    stdout = io.StringIO()
    sys_stdout = sys.stdout
    result = None
    try:
        lines = script.strip().split("\n")
        while lines and not lines[-1].strip():
            lines.pop()
        last_line = lines[-1] if lines else ""
        body = "\n".join(lines[:-1])
        exec_globals = globals().copy()
        exec_globals.update({'pd': pd, 'np': np})
        sys.stdout = stdout
        if body.strip():
            exec(body, exec_globals, local_vars)
        try:
            result = eval(last_line, exec_globals, local_vars)
        except Exception:
            exec(last_line, exec_globals, local_vars)
            result = None
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
                elif idx == 0 and isinstance(result, pd.DataFrame):
                    val = result
                if val is not None:
                    memory[name] = val
                    saved_vars.append(name)
                    if isinstance(val, (pd.DataFrame, pd.Series)):
                        _dataframes[name] = val
            if saved_vars:
                _notes.append(f"Saved to memory: {', '.join(saved_vars)}")
        output_text = stdout.getvalue().strip()
        if result is not None:
            output = repr(result)
        elif output_text:
            output = output_text
        else:
            if save_to_memory and saved_vars:
                output = f"Script executed successfully. Saved to memory: {', '.join(saved_vars)}"
            else:
                output = "Script executed successfully (no output)"
    except Exception as e:
        tb = traceback.format_exc()
        output = f"Error: {str(e)}\n{tb}"
    finally:
        sys.stdout = sys_stdout
    _notes.append(f"Script executed: {output[:100]}...")
    # After execution, log the exploration step
    _log_exploration_step(
        step_type="script_execution",
        operation=script[:100] + "..." if len(script) > 100 else script,
        result="Script executed successfully"
    )
    return [TextContent(type="text", text=output)]
@mcp.tool()
def list_dataframes() -> list:
    """List all DataFrames currently loaded in memory."""
    global _dataframes
    if not _dataframes:
        return [TextContent(type="text", text="No DataFrames loaded. Use run_sql_query_enhanced tool first.")]
    
    result = "=== DATAFRAMES IN MEMORY ===\n\n"
    for name, df in _dataframes.items():
        result += f"DataFrame: {name}\n"
        result += f"  Shape: {df.shape}\n"
        result += f"  Columns: {list(df.columns)}\n\n"
    
    result += f"Use these names in scripts: {', '.join(_dataframes.keys())}"
    return [TextContent(type="text", text=result)]



@mcp.tool()

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

### Data Exploration Tools Description & Schema
_dataframes: Dict[str, pd.DataFrame] = {}

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


### Prompt templates
class DataExplorationPrompts(str, Enum):
    EXPLORE_DATA = "explore-data"

class PromptArgs(str, Enum):
    CSV_PATH = "csv_path"
    TOPIC = "topic"

# --- Prompt template for database chat/analysis ---
DB_PROMPT_TEMPLATE = """
You are a professional Data Scientist and SQL expert. You have access to a MySQL database connection and a set of tools for querying, exploring, and visualizing data.

## Your Task:
1. Explore the database schema (tables and columns) using the `list_db_tables` tool.
2. For any analysis, always:
   - Use `run_sql_query_enhanced` to run SQL queries and load results into DataFrames.
   - Use `list_dataframes` to see available DataFrames and their columns.
   - Use `create_visualization` to generate charts from DataFrames.
3. When asked a question, follow this process:
   - Identify which tables and columns are relevant.
   - Write a SQL query to answer the question (limit results if needed).
   - Load the query result into a DataFrame using `run_sql_query_enhanced`.
   - Analyze the DataFrame and, if appropriate, create a visualization.
   - Summarize your findings concisely.

## Important Guidelines:
- Always use the tools (`list_db_tables`, `run_sql_query_enhanced`, `list_dataframes`, `create_visualization`) for all data access and analysis.
- Never access the database directly except through the provided tools.
- Limit query result sizes to avoid large outputs (e.g., use LIMIT or aggregation).
- If you are unsure about the schema, call `list_db_tables` first.
- Provide SQL queries that are safe and efficient.

Begin by exploring the database schema and proceed step by step for each user question or analysis request.
"""

#    d. Render the results returned by the run_script tool as a chart using plotly.js (prefer loading from cdnjs.cloudflare.com). Do not use react or recharts, and do not read the original CSV file directly. Provide the plotly.js code to generate the chart.


class DataExplorationTools(str, Enum):

    LOAD_CSV = "load_csv"
    RUN_SCRIPT = "run_script"


@mcp.prompt()
def explore_data_prompt():
    """A prompt to explore a  database as a data scientist"""
    print("Registering explore_data_prompt")
    return DB_PROMPT_TEMPLATE


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
# Resource: Supported chart types
@mcp.resource("data-exploration://chart-types", mime_type="application/json")
def chart_types_resource():
    """List of supported chart types for visualization."""
    print("[RESOURCE] chart_types_resource called")
    return [
        "histogram",
        "line",
        "bar",
        "pie",
        "area",
        "scatter"
    ]
# Keep notes resource as well
@mcp.resource("data-exploration://notes", mime_type="text/plain")
def notes_resource():
    """Notes generated by the data exploration server."""
    print("[RESOURCE] notes_resource called")
    global _notes
    return "\n".join(_notes) if _notes else "No notes yet. Use load_csv or run_script to generate notes."

if __name__ == "__main__":
    # --- Connect to MySQL and share the engine globally ---
    try:
        from sqlalchemy import create_engine
        from dotenv import load_dotenv
        import os
        load_dotenv()  # Load environment variables from .env file
        MYSQL_USER = os.getenv("MYSQL_USER")
        MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
        MYSQL_HOST = os.getenv("MYSQL_HOST")
        MYSQL_PORT = os.getenv("MYSQL_PORT")
        MYSQL_DB = os.getenv("MYSQL_DB")

        db_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
        print(f"[Startup] Initializing MySQL engine with URL: {db_url}")
        db_engine = create_engine(db_url)
        print("[Startup] MySQL engine initialized and shared globally.")
    except Exception as e:
        db_engine = None
        print(f"[Startup] MySQL engine initialization failed: {e}")

    mcp.run(transport="streamable-http")
