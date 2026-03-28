"""
基于streamlit框架完成WEB网页上传
"""
import streamlit as st
st.title("知识库更新服务")

upload_file=st.file_uploader(
    label="请上传txt文件",
    type="txt",
    accept_multiple_files=False #不接受多文件上传
)

if upload_file is not None:
    file_name = upload_file.name
    file_type = upload_file.type
    file_size = round(upload_file.size / 1024,2)  #KB
    st.subheader(f"文件名{file_name}")
    st.write(f"文件格式{file_type} | {file_size} MB")

    text=upload_file.getvalue().decode("utf-8")
    st.write(text)