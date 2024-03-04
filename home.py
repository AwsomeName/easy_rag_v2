import streamlit as st

session_stats = st.session_state


st.set_page_config(
    page_title="你好",
    page_icon="👋",
)

st.write("# 欢迎! 👋")

st.markdown(
    """
    这是一个基于大模型的文档问答框架。
    **👈 从侧边栏选择一个功能，开始大模型问答之旅吧！
    1. file QA, 支持选择、上传文件，进行问答。
    2. folder_QA, 读取一个文件夹内的所有文件，进行搜索问答。
"""
)
