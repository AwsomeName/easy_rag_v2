from fastapi import FastAPI, Request
import uvicorn, json
from rag_faiss_llama_index import RAGStringQueryEngine

app = FastAPI()

@app.post("/api/v2/file_query")
async def create_item(request: Request):
    # global kl
    global rag
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    input_str = json_post_list.get('input_str')
    file_path = json_post_list.get('file_path')
    # output_str, output_df = kl.search_result(input_str)
    output_str = rag.file_query(file_path, input_str)

    return output_str

    
if __name__ == '__main__':
    rag = RAGStringQueryEngine()
    uvicorn.run(app, host='0.0.0.0', port=11073, workers=1)