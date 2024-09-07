import torch
import os, faiss
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate
)
import requests
import json
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.embeddings import resolve_embed_model
from transformers import AutoTokenizer, AutoModel

class RAGStringQueryEngine():
    def __init__(
            self,
            emb_path: str = "local:/home/lc/models/BAAI/bge-small-zh-v1.5",
            faiss_demension: int = 512,
            # PERSIST_DIR: str = "./storage",
            ) -> None:
        
        self.emb_path = emb_path
        self.faiss_demension = faiss_demension
        Settings.embed_model = resolve_embed_model(self.emb_path)
        Settings.chunk_size = 30000
        
    def req_llm(self, query_str):
        url = "http://127.0.0.1:3332/v1/chat/completions"
        payload = json.dumps({
            "model": "qwen",
            "messages": [{
              "role": "user",
              "content": query_str
            }],
            "repetition_penalty": 1.4
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.text)
        infos = json.loads(response.text)['choices'][0]['message']['content']
        print(infos)
        return infos


    def query_raw(self, query_str: str):
        PERSIST_DIR: str = "./tmp_storage",
        self.PERSIST_DIR = PERSIST_DIR
        self.data_path = "/home/lc/code/easy_rag_v2/data_doc"
        
        print("loading index ...")

        if not os.path.exists(self.PERSIST_DIR):
            faiss_index = faiss.IndexFlatL2(self.faiss_demension)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
            # load the documents and create the index
            filename_fn = lambda filename: {"file_name": filename}
            docs = SimpleDirectoryReader(self.data_path, file_metadata=filename_fn).load_data()
            for d in docs:
                print("d:", d.metadata, d.node_id)
            self.index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
            # store it for later
            self.index.storage_context.persist(persist_dir=PERSIST_DIR)
            self.retriever = self.index.as_retriever()
        else:
            # load the existing index
            vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=PERSIST_DIR)
            self.index = load_index_from_storage(storage_context=storage_context)
            self.retriever = self.index.as_retriever()

        nodes = self.retriever.retrieve(query_str)
        # for n in nodes:
        #     print("------")
        #     print(n.node_id)
        #     print(n.metadata)
        #     print(n.score)
        #     print(n.text)
        
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
        如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
        问题:
        {question}
        已知内容:
        {context}
        
        """
        text2chatglm = prompt_template.format_map({
            'question': query_str,
            'context': context_str
        })
        print("len: ", len(text2chatglm))
        resp = self.req_llm(text2chatglm)
        print(resp)
        return resp

        
    def file_query(self, input_file, query_str, user="default"):
        print("input_file:", input_file)
        PERSIST_DIR = "./tempDir/" + user + "_" + input_file
        self.PERSIST_DIR = PERSIST_DIR
        self.data_path = "./" + input_file

        if not os.path.exists(PERSIST_DIR):
            faiss_index = faiss.IndexFlatL2(self.faiss_demension)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
            # load the documents and create the index
            filename_fn = lambda filename: {"file_name": filename}
            docs = SimpleDirectoryReader(self.data_path, file_metadata=filename_fn).load_data()
            f_docs = [d for d in docs if d.get_content() != None]
            for d in f_docs:
                print("d:", d.metadata, d.node_id)
                print(d.get_content())
            self.index = VectorStoreIndex.from_documents(f_docs, storage_context=storage_context)
            # store it for later
            self.index.storage_context.persist(persist_dir=PERSIST_DIR)
            self.retriever = self.index.as_retriever()
        else:
            # load the existing index
            vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=PERSIST_DIR)
            self.index = load_index_from_storage(storage_context=storage_context)
            self.retriever = self.index.as_retriever()

        nodes = self.retriever.retrieve(query_str)
        
        refs = [{"分数": str(n.score)[:5], "文件": str(n.metadata), "文本": n.text[:300]} for n in nodes]
        
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        
        prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
        如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
        问题:
        {question}
        已知内容:
        {context}
        
        """
        text2chatglm = prompt_template.format_map({
            'question': query_str,
            'context': context_str
        })
        # print("input: ", text2chatglm)
        print("len: ", len(text2chatglm))
        resp = self.req_llm(text2chatglm)
        print("resp:", resp)
        return {"resp": resp, "refs": refs}

    def file_summary(self, input_file, query_str, user="default"):
        # PERSIST_DIR = "./tempDir/" + user + "_" + input_file
        # self.PERSIST_DIR = PERSIST_DIR
        self.data_path = "./" + input_file

        filename_fn = lambda filename: {"file_name": filename}
        docs = SimpleDirectoryReader(self.data_path, file_metadata=filename_fn).load_data()
        content = ""
        for d in docs:
            print("d:", d.metadata, d.node_id)
            content += d.get_content()
        
        
        prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
        如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
        问题:
        {question}
        已知内容:
        {context}
        
        请回答
        """
        text2chatglm = prompt_template.format_map({
            'question': query_str,
            'context': content
        })
        # print("input: ", text2chatglm)
        print("len: ", len(text2chatglm))
        resp = self.req_llm(text2chatglm)
        return {"resp": resp, "refs": []}

           
        
if __name__ == "__main__":
    test = RAGStringQueryEngine()
    test.query_raw("项目相关人员都有谁，列出联系方式")