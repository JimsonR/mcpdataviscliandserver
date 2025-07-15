import pandas as pd

class DataValidationAgent:
    def __init__(self, schema: dict):
        """
        schema: dict of column_name -> expected_type (e.g., {'age': int, 'name': str})
        """
        self.schema = schema
        self.errors = []

    def validate_schema(self, df: pd.DataFrame):
        for col, expected_type in self.schema.items():
            if col not in df.columns:
                self.errors.append(f"Missing column: {col}")
            elif not df[col].map(lambda x: isinstance(x, expected_type) or pd.isnull(x)).all():
                self.errors.append(f"Column {col} has invalid types.")

    def validate_missing(self, df: pd.DataFrame):
        missing = df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                self.errors.append(f"Column {col} has {count} missing values.")

    def validate_ranges(self, df: pd.DataFrame, ranges: dict):
        for col, (min_val, max_val) in ranges.items():
            if col in df.columns:
                if not df[col].between(min_val, max_val).all():
                    self.errors.append(f"Column {col} has values outside range {min_val}-{max_val}.")

    def validate_uniqueness(self, df: pd.DataFrame, unique_cols: list):
        for col in unique_cols:
            if col in df.columns and not df[col].is_unique:
                self.errors.append(f"Column {col} has duplicate values.")

    def run_all(self, df: pd.DataFrame, ranges: dict = {}, unique_cols: list = []):
        self.errors = []
        self.validate_schema(df)
        self.validate_missing(df)
        if ranges:
            self.validate_ranges(df, ranges)
        if unique_cols:
            self.validate_uniqueness(df, unique_cols)
        return self.errors

# Example usage:
# schema = {'age': int, 'name': str, 'salary': float}
# ranges = {'age': (18, 99), 'salary': (0, 1_000_000)}
# unique_cols = ['id']
# df = pd.read_csv('your_data.csv')
# agent = DataValidationAgent(schema)
# errors = agent.run_all(df, ranges, unique_cols)
# if errors:
#     print("Validation errors:", errors)
# else:
#     print("Data is valid!")