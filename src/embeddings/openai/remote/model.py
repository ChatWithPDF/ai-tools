from request import ModelRequest
from openai import OpenAI
import os
import pandas as pd

class Model:
    embedding_model = "text-embedding-ada-002"

    def __new__(cls, context):
        cls.context = context
        if not hasattr(cls, 'instance'):
            cls.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY") 
            )
            cls.instance = super(Model, cls).__new__(cls)
        return cls.instance

    async def inference(self, request: ModelRequest):
        if request.df is not None:
            data = request.df
            data = data.loc[~pd.isnull(data['content']),:]
            data['content'] = data['content'].astype(str)

            if data.empty or data['content'].isnull().any():
                return 'There are nonzero null rows'

            data['embeddings'] = data['content'].apply(
                lambda x: self.client.embeddings.create(
                    input=x,
                    model=self.embedding_model,
                ).data[0].embedding
            )
            csv_string = data.to_csv(index=False)
            return str(csv_string)

        if request.query is not None:
            embedding = self.client.embeddings.create(
                input=request.query,
                model=self.embedding_model,
            ).data[0].embedding
            return [embedding]

        return "Invalid input"
