import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from feature_engine.encoding import RareLabelEncoder


@st.cache(allow_output_mutation=True)
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file).astype(np.float64,errors="ignore")


def Gorsellestir(degisken1,hedefDegisken,lineSelect):
    degisken = degisken1 + "?"
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    df[degisken] = df[degisken].astype("object").fillna("NaN")
    barChartDf = pd.DataFrame(df.groupby([degisken],dropna=False).size(),columns=["Toplam"]).reset_index()
    barChartDf["Yüzde"] = barChartDf.Toplam / barChartDf.Toplam.sum()
    
    figBar = px.bar(barChartDf, x=degisken, y="Yüzde",
                    color_discrete_sequence=px.colors.qualitative.Set2,hover_data=[barChartDf.Toplam],
                    text="Toplam")
    
    
    x = df.groupby([degisken,hedefDegisken]).size()
    y = df.groupby([degisken]).size()

    lineChartDf = pd.DataFrame(df.groupby([degisken,hedefDegisken]).size(),columns=["Yüzde"]).reset_index()
    lineChartDf["Yüzde"] = (x/y).values
    lineChartDf["Yüzde"] = lineChartDf["Yüzde"].fillna(0)
    

    
    figLine = px.line(lineChartDf,x=degisken,y="Yüzde",color=hedefDegisken,
                      color_discrete_sequence=px.colors.qualitative.Dark2)
    

    

    lineIndex = np.ravel([np.where(lineChartDf[hedefDegisken].unique() == lineSelect)])
    
    for figdata in lineIndex:
        fig.add_trace(figLine["data"][figdata],secondary_y=True)

    for figdata in range(len(figBar["data"])):
        fig.add_trace(figBar["data"][figdata])

    fig.update_layout(barmode="group" , title_text="{}-{}".format(degisken1,hedefDegisken) ,
                      legend_title_text=hedefDegisken , yaxis_tickformat = '.1%', yaxis2_tickformat = '.1%',
                      yaxis_title="Toplam Oran" , yaxis2_title="Oran" , xaxis_title="{}".format(degisken1))
    
    st.write(fig)
    

def CategoricProcessing(degisken1,tol):
    enc = RareLabelEncoder(tol=tol, n_categories=0, max_n_categories=None, replace_with='Others',variables=None, ignore_format=True)
    df[degisken1+"?"] = enc.fit_transform(df[[degisken1]].fillna("NA"))


def nonCategoricProcessing(degisken1,q):
    _, edges = pd.cut(df[degisken1], bins=q, retbins=True , right=False)
    labels = [f'{abs(edges[i]):.2f}-{edges[i+1]:.2f}' for i in range(len(edges)-1)]
    seri = pd.cut(df[degisken1] , bins=q,labels=labels)
    df[degisken1+"?"] = seri.apply(lambda x:str(x).replace("(","").replace("]","").replace(", ","-"))




def VeriGorsellestirme(degisken1,hedefDegisken,q,tol,line):
    if degisken1 == "Seç" or hedefDegisken == "Seç":
        return None
    
    
    if df[degisken1].dtype == "object":
        CategoricProcessing(degisken1,tol)
    else:
        if len(df[degisken1].unique()) <= 10:
            CategoricProcessing(degisken1,tol)
        else:
            nonCategoricProcessing(degisken1,q)


    Gorsellestir(degisken1,hedefDegisken,line)

        
    try:
        df.drop(columns=["{}?".format(degisken1)],inplace=True)
    except:
        pass










#streamlit run VeriAnaliziFinal.py --server.maxUploadSize=1028
st.sidebar.write("Made by Analytics Team")
st.sidebar.markdown('### **1️⃣ Csv Dosyası Seç 👇 **')
uploaded_file = st.sidebar.file_uploader("", type="csv", key='file_uploader')

if uploaded_file is not None:
    df = load_data(uploaded_file)  
    
    degisken1W = st.sidebar.selectbox(label="X değişkeni",options=["Seç"] + list(df.columns))
    hedefDegiskenW = st.sidebar.selectbox(label="Hedef Değişken",options=["Seç"] + list(df.columns))
    qW = st.sidebar.slider('Nümerik Değişken Dilim Sayısı', min_value=1, max_value=50 , value=5 , step=1, format="%d")
    tolW = st.sidebar.slider('Nümerik Değişken Dilim Sayısı', min_value=0.0, max_value=1.0 , value=0.05 , step=0.001, format="%f")
    
    if hedefDegiskenW == "Seç":
        lineW = st.sidebar.selectbox(label="Değişken Değerleri", options=["Seç"])
    else:
        lineW = st.sidebar.selectbox(label="Değişken Değerleri", options=["Seç"] + list(df[hedefDegiskenW].unique()) )
    
    

    VeriGorsellestirme(degisken1W,hedefDegiskenW,qW,tolW,lineW)
    
    
