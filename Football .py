#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[76]:


df = pd.read_csv('Dataset/results.csv')
df.head()


# # Deskripsi Variabel

# Berdasarkan informasi dari dataset tersebut hasil pertandingan bola dri tahun 1872-2022 (hingga piala dunia berlangsung) sebagai berikut :
# 1. date	: tangggal berlangsungnya pertandingan
# 2. home_team	: team tuan rumah yang bermain
# 3. away_team	: team tandang yang bermai
# 4. home_score	: skor team rumah
# 5. away_score	: skor team tandang
# 6. tournament	: nama turnamen/kompetisi yang tercatat oleh FIFA
# 7. city	: kota tempat match berlangsung
# 8. country	: negara tempat match berlangsung
# 9. neutral : tempat match berlangsung bersifat netral(bukan tuan rumah)

# In[3]:


df.describe().T


# Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:
# 
# * Count adalah jumlah sample pada data.
# * Mean adalah nilai rata-rata
# * Std adalah standar deviasi
# * Min yaitu nilai minimum setiap kolom
# * 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
# * 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
# * 75% adalah kuartil ketiga.
# * Max adalah nilai maksimum.

# In[77]:


df.info()


# Berdasarkan informasi dari dataset tersebut hasil pertandingan bola dri tahun 1872-2022 (hingga piala dunia berlangsung) sebagai berikut :
# 1. date	: berisi tanggal dan bersifat object (non-int)
# 2. home_team	: berisi nama tim yang menjadi tuan rumah (negara)
# 3. away_team	: berisi nama tim yang menjadi tandang (negara)
# 4. home_score	: skor bersifat float (1.0 , 2.0 , 3.0 dst)
# 5. away_score	: skor bersifat float (1.0 , 2.0 , 3.0 dst)
# 6. tournament	:  berisi nama turnamen/kompetisi yang tercatat oleh FIFA
# 7. city	:  berisi nama kota tempat match berlangsung
# 8. country	: berisi nama negara tempat match berlangsung
# 9. neutral : berisi sifat (False/True)

# In[4]:


df.isnull().sum()


# In[5]:


print(df[df['home_score'].isnull()])


# Berdasarkan EDA menunjukan bahwa masih ada data yang bersifat NaN ditunjukkan pada tournament yang berlangsung (FIFA World Cup) maka dihilangkan saja 4 data tersebut

# In[6]:


df=df.dropna()
df.isnull().sum()


# In[9]:


df.boxplot(column=['home_score','away_score'])


# Target analisis ini adalah menggunakan home dan away score mulai dari tahun 1872-2022 dan terlihat masih banyak outlier yang terdapat pada box plot diduga sepakbola sebelum jaman modern menghasilkan score diatas 5 dan untuk skor tuan rumah tertinggi mencapai 30 dan akan dinormaliasiskan saja dengan mencari skor dibawah 15 gol (asumsi pribadi)

# In[10]:


higher_home=15
higher_away=15
df=df[(df["home_score"]<higher_home) & (df["away_score"]<higher_away)]
df.head(5)


# In[11]:


conditions  = [df["home_score"] == df["away_score"], 
               df["home_score"] > df["away_score"] , 
               df["home_score"] < df["away_score"]]
choices     = [ "Draw", 'Win', 'Lost' ]
df["Keterangan"] = np.select(conditions, choices)
df.head(10)


# Kemudian, dengan berdasarkan score bisa kita klasifikasikan antara status kemenangan setiap match bergunan untuk kedepannya

# In[12]:


countries=df.home_team.unique()
print(f"There are {len(countries)} Countries in the home_team Column\n")
print(f"Countries-{countries}")


# Berdasarkan data tersebut, menunjukan negara yang tercata sebanyak 309 negara yang terlibat dan ada beberapa negara yang sudah tidak ada seperti German Dr ketika German barat dan German timur masih bertikai

# In[13]:


rank_bound = 10
ax = df.tournament.value_counts()[:rank_bound].sort_values()
value = ax.values
label = ax.index

plt.figure(figsize=(14,6))
plt.barh(y=label, width=value, edgecolor="k")
for i in range(rank_bound):
    plt.text(x=50,y=i-0.1,s=value[i],color="w",fontsize=12)
plt.show()


# Berdasarkan data tersebut, pertandingan  persahabatan mendominasi berdasarkan aturan FIFA dalam pemeringkatan negara bahwa pertangdingan persahabatan faktor pengali lebih kecil dibandingkan dengan major tournament dengan merujuk pada [ini](https://id.wikipedia.org/wiki/Peringkat_Dunia_FIFA)

# In[14]:


rank_bound = 10
ax = df.country.value_counts()[:rank_bound].sort_values()
value = ax.values
label = ax.index

plt.figure(figsize=(14,6))
plt.barh(y=label, width=value, edgecolor="k")
for i in range(rank_bound):
    plt.text(x=10,y=i-0.1,s=value[i],color="w",fontsize=12)
plt.show()


# In[15]:


years = []
for date in df.date:
    years.append(int(str(date)[0:4]))
plt.figure(figsize=(14,6))
plt.hist(years, density=True, bins=10, edgecolor="k")
plt.title("Histogram of Years")
plt.ylabel("Frequency")
plt.xlabel("Year")
plt.show()


# Setelah tahun 1960, frekuensi pertandingan meningkat tajam dikarenakan beberapa negara yang termasuk dalam region ataupun lainnya mengadakan turnamen skala major jadi berdampak tajam untuk pertandingan tersebut.

# In[16]:


data_home=df.loc[df["home_team"]==df["country"] ]
data_home=df.loc[df["tournament"] != "Friendly"]
data_home


# In[19]:


#Home team results
sns.displot(data_home, x="Keterangan")
plt.title("Home Team Winning Status")


# Berdasarkan ini tim tuan rumah memiliki tingkat kemenangan yang lebih tinggi dibandingkan kekalahan

# In[21]:


data_home['Keterangan'].value_counts()


# In[23]:


teams_win_statues=pd.crosstab(df["home_team"], 
                              df["Keterangan"],
                              margins=True, 
                              margins_name="Total")
teams_win_statues["team_win_probability"]=teams_win_statues["Win"]/(teams_win_statues["Total"])

#mencari total match home > 250 dengan rasio win tertinggi pada setiap negara
teams_win_statues_100=teams_win_statues.loc[teams_win_statues["Total"]>250]
teams_win_statues_100=teams_win_statues_100.sort_values("team_win_probability",ascending=False)
teams_win_statues_100.head(20)


# In[25]:


teams_away_statues=pd.crosstab(df["away_team"], 
                               df["Keterangan"],
                               margins=True, 
                               margins_name="Total")
teams_away_statues["team_win_probability"]=teams_away_statues["Lost"]/(teams_away_statues["Total"])

#mencari total match away > 250 dengan rasio win tertinggi pada setiap negara
teams_away_statues_100=teams_away_statues.loc[teams_away_statues["Total"]>250]
teams_away_statues_100=teams_away_statues_100.sort_values("team_win_probability",ascending=False)

teams_away_statues_100.rename(columns={'Lost': 'Win', 'Win' : 'Lost'}, 
                              index={'Win': 'Lost'}, 
                              inplace=True)
teams_away_statues_100.head(20)


# Membuat ML model untuk memprediksi suatu match berdasarkan data home & away score team (1872-2022)

# In[26]:


df_match = df.copy()
df_match.head(10)


# In[27]:


New_Dataset_part_1=pd.DataFrame(list(zip(years,df_match.values[:,7],
                                         df_match.values[:,1],
                                         df_match.values[:,2],
                                         df_match.values[:,3],
                                         df_match.values[:,4])),
                                columns=["year","Country","team_1",
                                         "team_2","team_1_score","team_2_score"])
#Buat dataset kedua yang berdasarkan hasil skor
New_Dataset_part_2=pd.DataFrame(list(zip(years,
                                         df_match.values[:,7],df_match.values[:,2],
                                         df_match.values[:,1],df_match.values[:,4],
                                         df_match.values[:,3])),
                                columns=["year","Country","team_1","team_2","team_1_score","team_2_score"])
New_Dataset=pd.concat([New_Dataset_part_1,New_Dataset_part_2],axis=0)
New_Dataset =New_Dataset.sample(frac=1).reset_index(drop=True) #Shaffling the dataset
New_Dataset.head(5)


# In[28]:


teams_1=New_Dataset.team_1.unique()
contries=New_Dataset.Country.unique()
all_countries=np.unique(np.concatenate((teams_1,contries), axis=0))
len(all_countries)


# In[31]:


sns.heatmap(New_Dataset.corr(),cmap='coolwarm', annot=True, fmt='.1f', linewidths=.1)
New_Dataset.corr()


# Melihat korelasi dari heatmap menunjukan korelasi sedikit kuat untuk dipertimbangkan target output berdasakrn skor home (team 1) dengan away (team 2)

# # Membuat model ML

# # Preprocessing dengan melabeli dataset

# In[46]:


# Defining the features and labels(Targets)

Y= New_Dataset.iloc[:,4:6] #Training targets (team_1_score and team_2_score)
categorized_data=New_Dataset.iloc[:,0:4].copy() #Traing features

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

#Labeling the data using LabelEncorder in Sklearn-(Giving a unique number to each string(country))

label_encoder.fit(all_countries)
#list(label_encoder.classes_)
categorized_data['team_1']=label_encoder.transform(categorized_data['team_1'])
categorized_data['team_2']=label_encoder.transform(categorized_data['team_2'])
categorized_data['Country']=label_encoder.transform(categorized_data['Country'])

#Converting these feature columns to categrize form to make the training processs more smoother
categorized_data['team_1']=categorized_data['team_1'].astype("category")
categorized_data['team_2']=categorized_data['team_2'].astype("category")
categorized_data['Country']=categorized_data['team_2'].astype("category")


# In[35]:


#Input Fatures to the model (x)

categorized_data.head(5)


# In[36]:


len(categorized_data)


# In[37]:


#Targets to the model (Y)

Y.head(5)


# In[38]:


len(Y)


# In[39]:


print(categorized_data.info())
print(Y.info())


# Membagi dataset dengan train test split untuk melihat algoritma ML yang mana cocok nantinya untuk digunakan 

# In[47]:


X=categorized_data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8,random_state = 123)


# In[41]:


from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[43]:


from sklearn.metrics import mean_squared_error
list_mse = []
for k in range(1, 21):
  model_knn = MultiOutputRegressor(KNeighborsClassifier(n_neighbors=k)).fit(X_train, y_train)
  test_mse = mean_squared_error(y_test, model_knn.predict(X_test))
  list_mse.append(test_mse)
  print(f"Nilai MSE untuk k = {k} adalah : {test_mse}")


# In[44]:


pd.DataFrame(list_mse, index=range(1, 21)).plot(
    xlabel="K",
    ylabel="MSE",
    legend=False,
    xticks=range(1,21), 
    figsize=(12,4),
    title='Visualisasi Nilai K terhadap MSE')


# In[45]:


df_models = pd.DataFrame(index=['Train MSE', 'Test MSE'], 
                      columns=['KNN', 'RandomForest'])


# In[48]:


KNN = MultiOutputRegressor(KNeighborsClassifier(n_neighbors=2)).fit(X_train, y_train)
df_models.loc['Train MSE', 'KNN'] = mean_squared_error(
    y_pred=KNN.predict(X_train),
    y_true=y_train)
df_models.loc['Test MSE', 'KNN'] = mean_squared_error(
    y_pred=KNN.predict(X_test),
    y_true=y_test)


# In[49]:


RF = MultiOutputRegressor(RandomForestClassifier()).fit(X_train,y_train)
df_models.loc['Train MSE', 'RandomForest'] = mean_squared_error(
    y_pred=RF.predict(X_train),
    y_true=y_train)
df_models.loc['Test MSE', 'RandomForest'] = mean_squared_error(
    y_pred=RF.predict(X_test),
    y_true=y_test)


# In[50]:


df_models


# In[51]:


fig, ax = plt.subplots()
df_models.T.sort_values(by='Test MSE', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
     


# Lakukan Predict dengan full data X dan Y setelah diuji data train dan test terlihat bahwa algrotima Random Forest lebih baik dibandingkan dengan KNN

# In[52]:


RF = MultiOutputRegressor(RandomForestClassifier()).fit(X,Y)


# In[53]:


prd=RF.predict(X)
prd


# In[54]:


score_team_1=[i[0] for i in prd]
score_team_2=[i[1] for i in prd]

from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(list(Y.iloc[:,0]),score_team_1)
cm2=confusion_matrix(list(Y.iloc[:,1]),score_team_2)


# In[58]:


plt.figure(figsize=(20,20))
sns.heatmap(cm1, annot=True,fmt="d", cmap='YlGnBu', linecolor='black', linewidths=1)
plt.title("Confusion Matrix for Team 1 Score")
plt.xlabel("Actual")
plt.ylabel("Predicted")


# In[56]:


from sklearn.metrics import classification_report
report_1=classification_report(Y.iloc[:,0],score_team_1)
print(report_1)


# In[57]:


plt.figure(figsize=(20,20))
sns.heatmap(cm2, annot=True,fmt="d", cmap='YlGnBu', linecolor='black', linewidths=1)
plt.title("Confusion Matrix for team 2 score")
plt.xlabel("Actual")
plt.ylabel("Predicted")


# In[59]:


report_2=classification_report(Y.iloc[:,1],score_team_2)
print(report_2)


# In[60]:


def select_winning_team(probability_array):
    prob_lst=[round(probability_array[0][i],3) for i in range(2)]
    if (prob_lst[0]>prob_lst[1]):
        out=0
    elif (prob_lst[0]<prob_lst[1]):
        out=1
    elif (prob_lst[0]==prob_lst[1]):
        out=2
    return out,prob_lst


# In[82]:


match_played=2022
team_1="United States"
team_2="Netherlands"
stadium="Qatar"

team_lst=[team_1,team_2]
team_1_num=label_encoder.transform([team_1])[0]
team_2_num=label_encoder.transform([team_2])[0]
stadium_num=label_encoder.transform([stadium])[0]

print(f"Team 01 is {team_1} -{team_1_num}")
print(f"Team 02 is {team_2} -{team_2_num}")
print(f"Played in  {stadium} -{stadium_num}")


# In[83]:


#Sample Prediction Output

X_feature=np.array([[match_played,stadium_num,team_1_num,team_2_num]])
res=RF.predict(X_feature)
win,_=select_winning_team(res)
try:
    print(f"{team_1} vs {team_2} \n {team_lst[win]} wins 🏆⚽🎯\n")
except IndexError:
    print(f"{team_1} vs {team_2} \n  Match Draw ⚽⚽⚽\n") 


# In[ ]:




