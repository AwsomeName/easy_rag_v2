import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import docx
import fitz

session_stats = st.session_state

st.set_page_config(
   page_title="文档搜索问答",
   page_icon="📝",
   layout="wide",
   initial_sidebar_state="expanded",
)

df = pd.DataFrame(
   np.random.randn(50, 20),
   columns=('col %d' % i for i in range(20)))

def save_uploaded_file(uploaded_file, save_directory, file_path):
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # st.success(f"已保存文件：{uploaded_file.name} 在 {save_directory} 中")

headers = {
    "Content-Type": "application/json"
}
save_directory = "tempDir"

if True:
    st.header("👇选择上传的文件")
    new_upload_file = st.file_uploader("", type=['docx', 'pdf', "txt"])
    print(new_upload_file)
    
    upload_file = None
    if new_upload_file:
        # 指定保存文件的目录（例如 tempDir）
        file_path = os.path.join(save_directory, new_upload_file.name)
        save_uploaded_file(new_upload_file, save_directory, file_path)


hcol1, hcol2 = st.columns(2)
with hcol1:
    input_str = st.text_input(label="👇在这里输入问题", placeholder="输入想要提问的内容", max_chars=400)
    
    if st.button("提问！"):
        if new_upload_file is not None:
            if input_str is not None and len(input_str) >0:
                with st.expander(label="生成结果", expanded=True):
                    with st.empty():
                        url = "http://0.0.0.0:11073/api/v2/file_query"
                        data = {'file_path': save_directory, 'input_str': input_str}
                        print("data:", data)
                        response = requests.post(url, json=data, headers=headers)
                        resp = response.json()["resp"]
                        print(resp)
                        st.markdown(resp)
        else:
            st.markdown("请选择文件")

with hcol2:
    st.header("参考依据")
    if new_upload_file:
        file_name = new_upload_file.name
        if file_name[-4:] == "docx":
            doc = docx.Document(new_upload_file)
            for para in doc.paragraphs:
                st.write(para.text)
        elif file_name[-3:] == "pdf":
            pdf_document = fitz.open(stream=new_upload_file.read(), filetype="pdf")
            # pdf_document = fitz.open(new_upload_file)
            # 展示 PDF 页面
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                st.image(page.get_pixmap(), caption=f"Page {page_num + 1}", use_column_width=True)
        else:
            file_type = "txt"
            
        
            
        
    
    # if st.session_state.get('output_df') is not None:
    #     st.dataframe(st.session_state.get('output_df'))  
    else:
        st.markdown("""
        ## 说明：
        1. 在正上方选择文件
        2. 在左上角输入`问题`，然后点击提问摁钮.
        3. 右上角会有`running`字样，表示程序正在运行.
        4. 结束后，会出现文本提取结果和对应的参考依据.
            - 4.1. 左下角文本框是生成的文本.
            - 4.2. 右侧是文本生成所参考的文档.
        """)

# with col2:
    # pass