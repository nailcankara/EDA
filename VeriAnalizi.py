import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots


class KesifselVeriAnalizi():
    
    def __init__(self,):
        self.df = None
        self.color1 = px.colors.qualitative.Set2
        self.color2 = px.colors.qualitative.Dark2 
    
    
    def getDF(self,):
        try:
            return self.df
        except:
            return None
    
    
    def dosyaYuklemeEkrani(self,):
        self.dfupload = st.sidebar.radio('', options=["Dosyadan Aktar","Url'den DF Aktar",
                                                      "Url'den Train-Test Aktar","Titanik"])
        
        nrows = 10000
        st.subheader("Not: En fazla {} satır veri okur.".format(nrows))
        
        if self.dfupload == "Dosyadan Aktar":
            try:
                csvfile = st.sidebar.file_uploader(label="Csv Aktar",type=["csv"])
                self.df = pd.read_csv(csvfile , nrows=nrows)
            except:
                pass
        elif self.dfupload == "Url'den DF Aktar":
            try:
                csvfile = st.sidebar.text_input("DF Url'si Girin")
                self.df = pd.read_csv(str(csvfile) , nrows=nrows)
            except:
                pass  
        elif self.dfupload == "Url'den Train-Test Aktar":
            try:
                trainURL = st.sidebar.text_input("Train Url'si Girin")
                testURL = st.sidebar.text_input("Test Url'si Girin")
                train = pd.read_csv(str(trainURL))
                test = pd.read_csv(str(testURL))
                self.df = train.append(test, ignore_index=True).iloc[:nrows,:]
            except:
                pass
        elif self.dfupload =="Titanik":
            try:
                trainURL= "http://cooltiming.com/SV/train.csv"
                testURL = "http://cooltiming.com/SV/test.csv"
                train = pd.read_csv(trainURL)
                test = pd.read_csv(testURL)
                self.df = train.append(test, ignore_index=True).iloc[:nrows,:]
            except:
                pass
        
        
        
    
    
    def veriTuruTespit(self,):
        
        
        self.istenmeyenDegiskenler = st.sidebar.multiselect("Görselleştirilmeyecek Değişkenler", 
                                                            options = self.df.columns)
        
        self.categorical_columns = []
        self.numerical_columns = []
        self.unnecessary_columns = self.istenmeyenDegiskenler
        
        for col in self.df.columns:
            if col in self.unnecessary_columns:
              continue  
            elif self.df[col].dtype == 'O':
                self.categorical_columns.append(col)
            else:
                self.numerical_columns.append(col)
    
    
    def RenkSecimi(self,):
        renksec = st.sidebar.checkbox(label="Grafik Rengini Değiştir (Opsiyonel)")
        if renksec:
            rengim = st.sidebar.selectbox(label="Renk Seç",
                                          options=["Renk1","Renk2"])
            
            if rengim == "Renk1":
                self.color1 = px.colors.qualitative.Set2
                self.color2 = px.colors.qualitative.Dark2
            elif rengim == "Renk2":
                self.color1 = px.colors.qualitative.Pastel1
                self.color2 = px.colors.qualitative.Set1

    
    def veriTuruAyarlama(self,):
        veriTuruDegis = st.sidebar.checkbox(label="Veri Türü Değiştir (Opsiyonel)")
        
        basarisizOperasyonlar = []
        if veriTuruDegis:
            for i in self.categorical_columns+self.numerical_columns:
                veriTuru = st.sidebar.selectbox(label="{}'nin veri türü:".format(i),
                                                options=["Seçim Yapma","Nümerik","Kategorik"])
                
                if veriTuru == "Nümerik":
                    try:
                        self.df[i] = self.df[i].astype(float)
                    except:
                        st.sidebar.text("{}: veri türü değiştirilemedi".format(i))
                elif veriTuru == "Kategorik":
                    try:
                        self.df[i] = self.df[i].astype(str)
                    except:
                        st.sidebar.text("{}: veri türü değiştirilemedi".format(i))
        
        self.categorical_columns = []
        self.numerical_columns = []
        
        for col in self.df.columns:
            if col in self.unnecessary_columns:
              continue  
            elif self.df[col].dtype == 'O':
                self.categorical_columns.append(col)
            else:
                self.numerical_columns.append(col)
                
                    
                
        
    
    def filtreIsteniyor(self,):
        filtrele = st.sidebar.checkbox("Filtrele")
        ilkKacSatir = st.slider("İlk Kaç Satır Gösterilsin",min_value=10,max_value=100,value=10,step=10)
        if filtrele:
            self.filtreEkrani()
            
        st.dataframe(self.df.head(ilkKacSatir))

    
    def filtreEkrani(self,):
        st.sidebar.title("Filtre Ekranı")

        
        cat_vals = []
        num_vals = []

        for i in range(len(self.categorical_columns)):

            
            val = st.sidebar.multiselect(self.categorical_columns[i],
                                         options=list(self.df[self.categorical_columns[i]].value_counts().keys()),
                                         default=list(self.df[self.categorical_columns[i]].value_counts().keys()))
            cat_vals.append(val)
        

        for i in range(len(self.numerical_columns)):
            val = st.sidebar.slider(self.numerical_columns[i],
                                    value=(self.df[self.numerical_columns[i]].min(),self.df[self.numerical_columns[i]].max()),
                                    min_value = self.df[self.numerical_columns[i]].min() , max_value=self.df[self.numerical_columns[i]].max())
            num_vals.append(val)
            

        for i,j in enumerate(cat_vals):
            self.df = self.df[self.df[self.categorical_columns[i]].isin(j)]

            
        for i,j in enumerate(num_vals):
            self.df =self.df[self.df[self.numerical_columns[i]] >= j[0]]
            self.df =self.df[self.df[self.numerical_columns[i]] <= j[1]]
            
    

    
    def getSonSutunlar(self,):
        return self.df.loc[:,self.categorical_columns+self.numerical_columns]
    
    
    def ozetIstatistik(self,):
        
        data = self.getSonSutunlar()
        col1 , col2 = st.beta_columns(2)
        col1.subheader("Veri Türleri")
        col2.subheader("Boş Değerler")
        veriTipi = pd.DataFrame(data.dtypes,columns=["Veri Tipi"]).replace(['object'],'Kategorik').replace(['float64'],'Nümerik').replace(["int64"],"Nümerik")
        col1.write(veriTipi)
        col2.write(pd.DataFrame(data.isnull().sum(),columns=["Na Sayısı"]))

        for i,j in enumerate(self.categorical_columns):
            if i%2==0:
                col1.subheader("{} için Veriler".format(j))
                col1.write(data[j].value_counts())
            else:
                col2.subheader("{} için Veriler".format(j))
                col2.write(data[j].value_counts())
            
        st.subheader("Tanımlayıcı İstatistik")
        st.write(data.describe())
    
    def ozetGrafik(self,):
        data = self.getSonSutunlar()
        for i in self.categorical_columns:
            fig = px.bar(data,
                         x=data[i].value_counts().keys(),
                         y=data[i].value_counts(),
                         color=data[i].value_counts().keys(),
                         text = ["%{}".format(round(j*100,2)) for j in data[i].value_counts()/data[i].value_counts().sum()],
                         title = "{}'nin BarChartı".format(i))
            st.write(fig)
            
            fig = px.pie(data,
                         names=data[i].value_counts().keys(),
                         values=data[i].value_counts(),
                         color=data[i].value_counts().keys(),
                         title = "{}'nin PieChartı".format(i))
            
            st.write(fig)
            
        for i in self.numerical_columns:
            bins = st.slider("{}'ın Dilim Sayısı".format(i), min_value=1, max_value=100,value=25)
            scale = st.checkbox("{} Log Scale".format(i))
            fig = px.histogram(data,
                               x=i,
                               nbins=bins,
                               marginal="box",
                               histfunc="count",title="{}'nin Histogramı".format(i),log_y=scale)
            st.write(fig)
        
    def histogram(self,):
        data = self.getSonSutunlar()
        target = self.hedef

        for i in self.numerical_columns:
            st.header("{}'ın Histogramı".format(i))
            bins = st.slider("{}'ın Dilim Sayısı".format(i), min_value=1, max_value=100,value=10)
            logScale = st.checkbox(label="LogScale {}".format(i))
  
            fig = px.histogram(data, x=i , nbins=bins , log_y=logScale)
            st.write(fig)
    
                      
        for i in self.categorical_columns:
            st.header("{}'ın BarChartı".format(i))
            logScale = st.checkbox(label="LogScale {}".format(i))
            
            fig = px.histogram(data, x=i, log_y=logScale)
            st.write(fig)            

            
            #col2.bar_chart(data[i])
    
    def korelasyon(self,):
        data = self.getSonSutunlar()
        
        fig = px.scatter_matrix(data,dimensions=self.getSonSutunlar(),color=self.hedef)
        fig.update_traces(diagonal_visible=False,showlowerhalf=False)
        st.write(fig)
        

    def hedefDegiskenGorsel(self,):
        self.hedef= st.selectbox(label="Hedef Değişken Seçimi",
                                         options=["Seç"]+self.numerical_columns+self.categorical_columns)
        
        ondalik = st.slider(label="Ondalık Gösterimi",min_value=0,max_value=3,step=1,value=2)
        data = self.getSonSutunlar()
        dataX = data.copy()
        st.title("GRAFİKLER")
        #long_df = px.data.medals_long()

        #fig4 = px.bar(long_df, x="nation", y="count", color="medal", title="Long-Form Input")
        
        #st.text(fig4)
        #st.write(fig4)
        
        
        if self.hedef in self.numerical_columns+self.categorical_columns:

            for i in self.numerical_columns:
                
                if i == self.hedef:
                    continue
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                bins = st.slider("{}'ın Dilim Sayısı".format(i), min_value=1, max_value=len(data[i].unique()) if 25 > len(data[i].unique()) else 50, value = len(data[i].unique()) if 25 > len(data[i].unique()) else 25  )
                _, edges = pd.cut(dataX[i], bins=bins, retbins=True)
                
                if ondalik==0:
                    labels = [f'{abs(edges[i]):.0f}-{edges[i+1]:.0f}' for i in range(len(edges)-1)]
                elif ondalik==1:
                    labels = [f'{abs(edges[i]):.1f}-{edges[i+1]:.1f}' for i in range(len(edges)-1)]
                elif ondalik==2:
                    labels = [f'{abs(edges[i]):.2f}-{edges[i+1]:.2f}' for i in range(len(edges)-1)]
                elif ondalik==3:
                    labels = [f'{abs(edges[i]):.3f}-{edges[i+1]:.3f}' for i in range(len(edges)-1)]

                dataX["AnlikSutun"] = pd.cut(dataX[i] , bins=bins,labels=labels)
                forbarChart = pd.DataFrame(dataX.groupby(["AnlikSutun",self.hedef]).size(),columns=["Count"])
                forbarChart["{}".format(i)] = forbarChart.index.get_level_values(0)
                forbarChart["{}?".format(self.hedef)] = forbarChart.index.get_level_values(1)
                forbarChart = forbarChart.reset_index(drop=True)
                forbarChart["Perc"] = forbarChart["Count"]/forbarChart["Count"].sum()
         
            
                emptySeri = pd.Series()
                for uni in forbarChart[i].unique():
                    Perc = forbarChart[forbarChart[i] == uni].Count / forbarChart[forbarChart[i] == uni].Count.sum()
                    emptySeri = emptySeri.append(Perc,ignore_index=True)
                forbarChart["Perc2"] = emptySeri
            
                figBar = px.bar(forbarChart,
                               x="{}".format(i),
                               y="Perc",
                               color=forbarChart["{}?".format(self.hedef)].astype(str),
                               color_discrete_sequence=self.color1)
            
                
                figLine = px.line(forbarChart,
                               x="{}".format(i),
                               y="Perc2",
                               color=forbarChart["{}?".format(self.hedef)].astype(str),
                               color_discrete_sequence=self.color2)
                
                
                lineSelection = st.multiselect(label="{} LineChart".format(i),
                                               options=forbarChart["{}?".format(self.hedef)].unique())
                
                lineIndex = []
                for line in lineSelection:
                    lineIndex.append(np.where(forbarChart["{}?".format(self.hedef)].unique() == line))
                lineIndex = np.ravel(lineIndex)

                for figdata in lineIndex:
                    fig.add_trace(figLine["data"][figdata],secondary_y=True)
                    
                for figdata in range(len(figBar["data"])):
                    fig.add_trace(figBar["data"][figdata])
                    
                fig.update_layout(barmode="relative" , title_text="{}-{}".format(i,self.hedef) ,
                                  legend_title_text=self.hedef , yaxis2_tickformat = '.1%', yaxis_tickformat='.0%',
                                  yaxis_title="Toplam Oran" , yaxis2_title="Oran" , xaxis_title="{}".format(i))
                        

                st.write(fig)
                

            for i in self.categorical_columns:
                if i == self.hedef:
                    continue

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                
                forbarChart = pd.DataFrame(data.groupby([i,self.hedef]).size(),columns=["Count"])
                forbarChart["{}".format(i)] = forbarChart.index.get_level_values(0)
                forbarChart["{}?".format(self.hedef)] = forbarChart.index.get_level_values(1)
                forbarChart = forbarChart.reset_index(drop=True)
                
                forbarChart["Perc"] = forbarChart["Count"]/forbarChart["Count"].sum()
                
                emptySeri = pd.Series()
                for uni in forbarChart[i].unique():
                    Perc = forbarChart[forbarChart[i] == uni].Count / forbarChart[forbarChart[i] == uni].Count.sum()
                    emptySeri = emptySeri.append(Perc,ignore_index=True)
                forbarChart["Perc2"] = emptySeri
                    

                
                figBar = px.bar(forbarChart,
                               x="{}".format(i),
                               y="Perc", #Count yapılabilir
                               color=forbarChart["{}?".format(self.hedef)].astype(str),
                               color_discrete_sequence=self.color1)
                
                figLine = px.line(forbarChart,
                               x="{}".format(i),
                               y="Perc2",
                               color=forbarChart["{}?".format(self.hedef)].astype(str),
                               color_discrete_sequence=self.color2)
                
                lineSelection = st.multiselect(label="{} LineChart".format(i),
                                               options=forbarChart["{}?".format(self.hedef)].unique())
                
                lineIndex = []
                for line in lineSelection:
                    lineIndex.append(np.where(forbarChart["{}?".format(self.hedef)].unique() == line))
                lineIndex = np.ravel(lineIndex)

                
                for figdata in lineIndex:
                    fig.add_trace(figLine["data"][figdata],secondary_y=True)
                
                
                for figdata in range(len(figBar["data"])):
                    fig.add_trace(figBar["data"][figdata])
                    
                
                    
                fig.update_layout(barmode="relative" , title_text="{}-{}".format(i,self.hedef) ,
                                  legend_title_text=self.hedef , yaxis2_tickformat = '.1%', yaxis_tickformat='.0%',
                                  yaxis_title="Toplam Oran" , yaxis2_title="Oran" , xaxis_title="{}".format(i))

            
                st.write(fig)
                  

            del dataX,forbarChart
                         
        

#st.set_page_config(layout="centered")

st.title('Interaktif Veri Analizi')
st.sidebar.header("Veri Yükleme Ekranı")

KVA = KesifselVeriAnalizi()
KVA.dosyaYuklemeEkrani()



if KVA.getDF() is not None:
    KVA.veriTuruTespit()
    KVA.RenkSecimi()
    KVA.veriTuruAyarlama() 
    KVA.filtreIsteniyor()

    
    col1,col2,col3 = st.beta_columns(3)
    
    ozetista = col1.checkbox(label="Özet İstatistik")
    ozetgraf = col2.checkbox(label="Özet Grafik")
    hedefDegisken = col3.checkbox(label="Hedef Değişken Görsel")
    #histogram = col3.checkbox(label="Dendogram")
    #korelasyon = col4.checkbox(label="Korelasyon")

    
    if ozetista:
        KVA.ozetIstatistik()
        
    if ozetgraf:
        KVA.ozetGrafik()
    
    if hedefDegisken:
        KVA.hedefDegiskenGorsel()
        
    #if histogram:
        #KVA.histogram()
    
    #if korelasyon:
        #KVA.korelasyon()
        
    
    
    
    
 
 