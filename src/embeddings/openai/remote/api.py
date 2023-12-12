from model import Model
from request import ModelRequest
from quart import Quart, request, send_file
import aiohttp
import pandas as pd
import io

app = Quart(__name__)

@app.before_serving
async def startup():
    app.client = aiohttp.ClientSession()

@app.route('/', methods=['POST'])
async def embed():
    files = await request.files
    uploaded_file = files.get('file')
    data = await request.get_json()
    model = Model(app)

    if uploaded_file:
        df = pd.read_csv(uploaded_file.stream)
        if df.empty or df['content'].isnull().any():
            return {'error': 'There are nonzero null rows'}, 400

        req = ModelRequest(df=df)
        response = await model.inference(req)

        if response == 'There are nonzero null rows':
            return {'error': response}, 400

        df = pd.read_csv(io.StringIO(response))
        df.to_csv('output.csv', index=False)
        return await send_file('output.csv', mimetype='text/csv', as_attachment=True, attachment_filename='output.csv')
    else:
        req = ModelRequest(**data)
        response = await model.inference(req)

        if response == 'There are nonzero null rows':
            return {'error': response}, 400

        return response
