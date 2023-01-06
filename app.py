import numpy as np
import altair as alt
import pandas as pd
import streamlit as st
import pymongo
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# model='./model/lenet.onnx'

# Initialize connection.
# Uses st.experimental_singleton to only run once.
@st.experimental_singleton
def init_connection():
    return pymongo.MongoClient(**st.secrets["mongo"])

client = init_connection()

# Pull data from the collection.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
@st.experimental_memo(ttl=600)
def get_data(model:str):
    # print(model)
    db = client["op_fusion"]
    res = {}
    fuse_time_item = db["fuse_time"].find_one({'name': model})
    res["fuse_time"]=fuse_time_item

    op_stats_item = db["op_stats"].find_one({'name': model})
    res["op_stats"]=op_stats_item

    exec_item = db["exec_time"].find_one({'name': model})
    res["exec_time"]=exec_item
    return res

# sidebar

st.sidebar.title("1. Data")

# Load platform
with st.sidebar.expander("Platform", expanded=True):
    platform_name = st.selectbox(
            "Select a platform",
            options=['MLIR','TVM'],
        )
    
# Load model
with st.sidebar.expander("Model", expanded=True):
    model_name = st.selectbox(
            "Select a Model",
            options=['lenet','resnet50-v1-12','bert-large-cased'],
        )
    model='./model/' + model_name + '.onnx'

st.sidebar.title("2. Evaluation")
st.sidebar.title("3. Forecast")
st.sidebar.title("4. Download")

# title

st.header(model_name + '  statistics')
st.write('Hello, *World!* :sunglasses:')


dicts = get_data(model)
# print(dicts)
# Print results.

fuse_time_dict=dicts['fuse_time']
op_stats_dict=dicts['op_stats']
exec_time=dicts['exec_time']
# print(fuse_time_dict)
# print(op_stats_dict)

# fuse_time

st.write('fuse_time')
fuse_time_df = pd.DataFrame({
     'op name': fuse_time_dict['lable'],
     'time': fuse_time_dict['value']
     })
st.dataframe(fuse_time_df, use_container_width=True) 
# st.write(fuse_time_df)
# df1 = fuse_time_df['time'].value_counts().rename_axis('unique_values').reset_index(name='counts')
# st.bar_chart(df1)
# st.bar_chart( sorted(fuse_time_dict['value'],reverse=True), x='op name', y='time/10^-5') 
st.bar_chart( fuse_time_dict, x='lable', y='value') 

# op_stats

st.write('op_stats')
op_stats_df = pd.DataFrame({
     'op name': op_stats_dict['lable'],
     'nums': op_stats_dict['value']
     })
st.dataframe(op_stats_df, use_container_width=True) 
# st.write(op_stats_df)
fig, ax = plt.subplots()
x=op_stats_dict['lable']
y= op_stats_dict['value']
explode = np.zeros(len(y))
explode[-1] = 0.1
patches, texts, autotexts = plt.pie(y,
        explode=explode,
        labels=x,
        autopct='%.2f%%', # 格式化输出百分比
        )
# 重新设置字体大小
proptease = fm.FontProperties()
proptease.set_size('xx-small')
# font size include: ‘xx-small’,x-small’,'small’,'medium’,‘large’,‘x-large’,‘xx-large’ or number, e.g. '12'
plt.setp(autotexts, fontproperties=proptease)
plt.setp(texts, fontproperties=proptease)
plt.title("op nums")

st.pyplot(fig)
# st.bar_chart( sorted(op_stats_dict['value'],reverse=True)) 

# exec

st.write('exec_time')
st.code(exec_time['exec'], language='shell')
# Example 5

