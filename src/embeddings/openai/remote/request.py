import json
import pandas as pd
from typing import Optional

class ModelRequest():
    def __init__(self, query: Optional[str] = None, df: Optional[pd.DataFrame] = None, query_type: Optional[str] = None):
    
        self.query = query
        self.query_type = query_type
        self.df = df

        # Additional validation or preprocessing can be added here

    def to_json(self) -> str:

        # Handle DataFrame serialization to JSON if necessary
        def default_serializer(o):
            if isinstance(o, pd.DataFrame):
                return o.to_dict(orient='records')
            return o.__dict__

        return json.dumps(self, default=default_serializer, sort_keys=True, indent=4)
