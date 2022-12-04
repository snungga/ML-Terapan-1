# ML-Terapan-1

# Laporan Proyek Machine Learning - Nungga Saputra

## 1. Domain Proyek

Proyek machine learning ini bertujuan untuk melihat seberapa berpengaruh track record dari hasil pertandingan dari tahun 1872-2022
- Tujuan dari ini adalah memprediksi suatu match seperti berlangsungnya pertandingan piala dunia (FIFA World Cup)

## 2. Business Understanding
Dengan mempertimbangkan match ataupun sejarah pertemuan dalam pertandingan antar negara dapat menentukan suatu "Big Match" berdasarkan home &away score.

### 2.1. Problem Statements
Poin-poin masalah yang terdapat dalam penetuan suatu match antara lain:
- Seberapa pengaruh dari banyaknya pertandingan suatu negara baik bertindak home ataupun away match untuk menentukan kemenangan?
- Berdasarkan dari score home & away match apakah dapat memprediksi suatu kemenangan team berdasarkan model machine learning dengan KNN atau Random Forest?

### 2.2. Goals
Tujuan dari implementasi solusi machine learning ini antara lain:
- Dengan melihat track record suatu match, dapat memutuskan negara mana akan yang memenangkan Major Tropy tanpa mempertimbangkan aspek lain seperti xG Goals, pemain yang cidera dsbnya.
- Membuat algoritma yang dapat memprediksi cepat tim mana yang menang dan kalah ataupun draw yang dapat membantu prediksi match berlangsung

### 2.3. Solution specifications
  Implementasi pemodelan *machine learning* untuk memprediksi suatu match kali ini memiliki spesifikasi sebagai berikut:
  
  - Mengimplementasikan pemodelan menggunakan dua buah algoritma: *KNeighbors*, *Random Forest* .
  
  - Penilaian performa terhadap ketiga pemodelan yang akan dibuat menggunakan beberapa buah metrik/metode pengukuran, antara lain sebagai berikut:

    a. **Mean Squared Error (MSE)** 
    
    - Metrik ini mengkuadratkan perbedaan nilai antara prediksi dan aktual, lalu mengambil nilai akhir rata-ratanya (Bickel, 2015).

       - Rumus MSE adalah sebagai berikut:

            $MSE = \frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$

    b. **Confusion Matrix** 
    
    - Matrix ini memetakan hasil prediksi ke dalam beberapa kategori, antara lain:
       
       |                | Nilai Prediksi | Nilai Aktual |
       |----------------|----------------|--------------|
       | True Positive  | 1              | 1            |
       | False Positive | 0              | 0            |
       | False Negative | 1              | 0            |
       | True Negative  | 0              | 1            |
       
    - Berikut ini adalah pemetaan dari confusion matrix:

       | Total Poulation    | (Predicted)  Positive | (Predicted)  Negative |
       |--------------------|-----------------------|-----------------------|
       | (Actual)  Positive | True Positive (TP)    | False Negative (FN)   |
       | (Actual)  Negative | False Positive (FP)   | True Negative (TN)    |


    - **Akurasi** 
       
       - Akurasi diukur dengan rumus berikut:

          $accuracy = \frac{(TP + TN)}{(TP + TN + FP + FN)}$
       
    - **Presisi** 
       
       - Presisi diukur dengan rumus berikut:
       
          $precision = \frac{TP}{(TP + FP)}$
       
    - **Sensitivitas / Recall** 
       
       - Sensitivitas diukur dengan rumus berikut:
       
          $sensitivity = \frac{TP}{(TP + FN)}$
          
          
## 3. Data Understanding
Data yang dipakai bersumber dari [kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

### 3.1. Variabel-variabel pada Dataset adalah sebagai berikut:
Berdasarkan informasi dari dataset tersebut hasil pertandingan bola dri tahun 1872-2022 (hingga piala dunia berlangsung) sebagai berikut :
1. date	: tangggal berlangsungnya pertandingan
2. home_team	: team tuan rumah yang bermain
3. away_team	: team tandang yang bermai
4. home_score	: skor team rumah
5. away_score	: skor team tandang
6. tournament	: nama turnamen/kompetisi yang tercatat oleh FIFA
7. city	: kota tempat match berlangsung
8. country	: negara tempat match berlangsung
9. neutral : tempat match berlangsung bersifat netral(bukan tuan rumah)

Setelah data mentah di-*load*, kita melakukan serangkaian aktivitas *exploratory* sebagai berikut:
- Melihat bagian awal tabel data dengan fungsi head().

| date       | home_team | away_team | home_score | away_score | tournament | city    | country  | neutral |
|------------|-----------|-----------|------------|------------|------------|---------|----------|---------|
| 1872-11-30 | Scotland  | England   | 0.0        | 0.0        | Friendly   | Glasgow | Scotland | False   |
| 1873-03-08 | England   | Scotland  | 4.0        | 2.0        | Friendly   | London  | England  | False   |
| 1874-03-07 | Scotland  | England   | 2.0        | 1.0        | Friendly   | Glasgow | Scotland | False   |
| 1875-03-06 | England   | Scotland  | 2.0        | 2.0        | Friendly   | London  | England  | False   |
| 1876-03-04 | Scotland  | England   | 3.0        | 0.0        | Friendly   | Glasgow | Scotland | False   |

- Melihat *summary* data dengan fungsi describe().

|           | count   |   mean   | std      | min | 25% | 50% | 75% | max  |
|------------|---------|----------|----------|-----|-----|-----|-----|------|
| home_score | 44202.0 | 1.739107 | 1.746388 | 0.0 | 1.0 | 1.0 | 2.0 | 31.0 |
| away_score | 44202.0 | 1.178069 | 1.394215 | 0.0 | 0.0 | 1.0 | 2.0 | 21.0 |

Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:

* Count adalah jumlah sample pada data.
* Mean adalah nilai rata-rata
* Std adalah standar deviasi
* Min yaitu nilai minimum setiap kolom
* 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
* 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
* 75% adalah kuartil ketiga.
* Max adalah nilai maksimum.

|#    |Column              |Non-Null Count    |Dtype  | 
|-----|--------------------|------------------|-------|  
|0    |date             |44206 non-null    |object | 
|1    |home_team              |44206 non-null    |object | 
|2    |away_team            |44206 non-null    |object | 
|3    |home_score          |44206 non-null    |float64 | 
|4    |away_score           |44206 non-null    |float64 | 
|5    |tournament     |44206 non-null    |object | 
|6    |city      |44206 non-null    |object |  
|7    |country   |44206 non-null    |object |
|8    |neutral          |44206 non-null    |bool |

Berdasarkan informasi dari dataset tersebut hasil pertandingan bola dri tahun 1872-2022 (hingga piala dunia berlangsung) sebagai berikut :
1. date	: berisi tanggal dan bersifat object (non-int)
2. home_team	: berisi nama tim yang menjadi tuan rumah (negara)
3. away_team	: berisi nama tim yang menjadi tandang (negara)
4. home_score	: skor bersifat float (1.0 , 2.0 , 3.0 dst)
5. away_score	: skor bersifat float (1.0 , 2.0 , 3.0 dst)
6. tournament	:  berisi nama turnamen/kompetisi yang tercatat oleh FIFA
7. city	:  berisi nama kota tempat match berlangsung
8. country	: berisi nama negara tempat match berlangsung
9. neutral : berisi sifat (False/True)

### 3.2. EDA (Exploration Data Analysis)

|[<img src="/image/1.png"/>](/image/1.png)|
|:--:| 
| *Gambar 1. Box Plot home dan away score.* |

Target analisis ini adalah menggunakan home dan away score mulai dari tahun 1872-2022 dan terlihat masih banyak outlier yang terdapat pada box plot diduga sepakbola sebelum jaman modern menghasilkan score diatas 5 dan untuk skor tuan rumah tertinggi mencapai 30 dan akan dinormaliasiskan saja dengan mencari skor dibawah 15 gol. Tabel dibawah juga menormalisasikan hasil pertandingan dengan penentuan keterangan Draw, Win, Lost. Berdasarkan data tersebut, menunjukan negara yang tercata sebanyak 309 negara yang terlibat dan ada beberapa negara yang sudah tidak ada seperti German Dr ketika German barat dan German timur masih bertikai.

| |date|home_team|away_team|home_score|away_score|tournament|city|country|neutral|Keterangan|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|0|1872-11-30|Scotland|England|0.0|0.0|Friendly|Glasgow|Scotland|False|Draw|
|1|1873-03-08|England|Scotland|4.0|2.0|Friendly|London|England|False|Win|
|2|1874-03-07|Scotland|England|2.0|1.0|Friendly|Glasgow|Scotland|False|Win|
|3|1875-03-06|England|Scotland|2.0|2.0|Friendly|London|England|False|Draw|
|4|1876-03-04|Scotland|England|3.0|0.0|Friendly|Glasgow|Scotland|False|Win|
|5|1876-03-25|Scotland|Wales|4.0|0.0|Friendly|Glasgow|Scotland|False|Win|
|6|1877-03-03|England|Scotland|1.0|3.0|Friendly|London|England|False|Lost|
|7|1877-03-05|Wales|Scotland|0.0|2.0|Friendly|Wrexham|Wales|False|Lost|
|8|1878-03-02|Scotland|England|7.0|2.0|Friendly|Glasgow|Scotland|False|Win|
|9|1878-03-23|Scotland|Wales|9.0|0.0|Friendly|Glasgow|Scotland|False|Win|

|[<img src="/image/2.png"/>](/image/2.png)|
|:--:| 
| *Gambar 2. Tipe Match paling banyak.* |

Berdasarkan data tersebut, pertandingan  persahabatan mendominasi berdasarkan aturan FIFA dalam pemeringkatan negara bahwa pertangdingan persahabatan faktor pengali lebih kecil dibandingkan dengan major tournament dengan merujuk pada [ini](https://id.wikipedia.org/wiki/Peringkat_Dunia_FIFA). Dengan demikian untuk dataset pertandingan persahabatan akan dihilangkan dengan lebih menitik beratkan pada Major Tournament.

|[<img src="/image/3.png"/>](/image/3.png)|
|:--:| 
| *Gambar 3. Negara dengan match paling banyak.* |

Dengan gambar tersebut, sebagai pertimbangan prediksi apakah mempengaruh atau tidak dengan banyaknya pertandingan pada saat prediksi suatu negara. 

|[<img src="/image/4.png"/>](/image/4.png)|
|:--:| 
| *Gambar 4. Histogram frekuensi pertandingan.* |

Setelah tahun 1960, frekuensi pertandingan meningkat tajam dikarenakan beberapa negara yang termasuk dalam region ataupun lainnya mengadakan turnamen skala major jadi berdampak tajam untuk pertandingan tersebut.

|Keterangan|Draw|Lost|Win|Total|team_win_probability| | | | | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Home_Team|
|Brazil|110|58|425|593|0.716695|
|Spain|71|51|257|379|0.678100|
|Argentina|124|69|374|567|0.659612|
|Egypt|69|74|260|403|0.645161|
|Ivory Coast|67|36|183|286|0.639860|
|Nigeria|69|36|184|289|0.636678|
|Iran|63|45|185|293|0.631399|
|Italy|123|52|291|466|0.624464|
|Germany|112|87|327|526|0.621673|
|England|115|83|325|523|0.621415|
|Morocco|71|47|193|311|0.620579|
|Ghana|74|40|186|300|0.620000|
|Russia|71|50|190|311|0.610932|
|Costa Rica|78|53|195|326|0.598160|
|South Korea|118|84|295|497|0.593561|
|Sweden|106|104|296|506|0.584980|
|Mexico|125|104|320|549|0.582878|
|Portugal|85|59|200|344|0.581395|
|France|102|109|292|503|0.580517|
|Netherlands|103|84|252|439|0.574032|

Kemudian dengan melihat rasio kemenangan home team , terlihat brazil mendominasi dengan 71% pertandingan dimenangkan sebagai team tuan rumah

|Keterangan|Draw|Win|Lost|Total|team_win_probability|
|:----|:----|:----|:----|:----|:----|
|away_team| | | | | |
|Germany|94|247|119|460|0.536957|
|Brazil|95|228|102|425|0.536471|
|England|136|271|117|524|0.517176|
|Spain|100|169|84|353|0.478754|
|South Korea|115|181|109|405|0.446914|
|Netherlands|83|169|129|381|0.443570|
|Russia|118|176|113|407|0.432432|
|Japan|57|117|103|277|0.422383|
|Italy|112|154|106|372|0.413978|
|Sweden|121|222|204|547|0.405850|
|Yugoslavia|61|115|115|291|0.395189|
|Mexico|89|148|140|377|0.392573|
|France|85|146|141|372|0.392473|
|Argentina|127|175|145|447|0.391499|
|Portugal|66|118|120|304|0.388158|
|Scotland|92|162|164|418|0.387560|
|Zambia|108|170|167|445|0.382022|
|Hungary|107|187|197|491|0.380855|
|Denmark|88|149|172|409|0.364303|
|Ivory Coast|84|110|109|303|0.363036|


Untuk Away team, Germany sebagai tertinggi dengan rasio 53% 

## 4. Data Preparation
Teknik *data preparation* yang dilakukan untuk mempersiapkan data sebelum diproses ke dalam model machine learning antara lain:

- **Menggabungkan dataset antara home & away score**
   yakni berdsarkan year, Country, team_1 (home), team_2 (away), team_1_score, team_2_score agar mempermudah untuk membagi dataset tersebut menjadi test dan train nantinya untuk output memprediksi pertandingan berdasarkan skor

   |[<img src="/image/6.png"/>](/image/6.png)|
   |:--:| 
   | *Gambar 6. Heatmap korelasi antara tahun bermain dengan skor masing-masing.* |
   berdasarkan gambar tersebut korelasi bernilai negatif dan sedikit adanya korelasi yang berkisar di 0.2 pada skor masing-masing

- **Data splitting**

  sebelum membagi dataset, dilakukan terlebih dahulu melabeling data dengan label encoder agar memperudah melakukan proses machine learning dalam bentuk numerik. Kemudian setelah datasset sudah dilabel akan dibagi menjadi data train dan tes dengan proporsional 80:20 dengan Y merupakan skor dan X berdasarkan *tim home & away, negara tempatnya match*

   |year|Country|team_1|team_2|
   |:----|:----|:----|:----|
   |0|2001|155|98|155|
   |1|1953|149|131|149|
   |2|1993|302|286|302|
   |3|2013|197|194|197|
   |4|1960|291|65|291|


## 5. Modeling
Pada tahap ini, akan menggunakan tiga algoritma untuk regresi. Kemudian, akan dilakukan evaluasi performa masing-masing algoritma dan menetukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan digunakan, antara lain:

1. K-Nearest Neighbor
Kelebihan algoritma KNN adalah mudah dipahami dan digunakan sedangkan kekurangannya jika dihadapkan pada jumlah fitur atau dimensi yang besar rawan terjadi bias.

2. Random Forest
Kelebihan algoritma Random Forest adalah menggunakan teknik Bagging yang berusaha melawan overfitting dengan berjalan secara paralel. Sedangkan kekurangannya ada pada kompleksitas algoritma Random Forest yang membutuhkan waktu relatif lebih lama dan daya komputasi yang lebih tinggi dibanding algoritma seperti Decision Tree.

### 5.1 KNN
   KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih k tetangga terdekat. Pemilihan nilai k sangat penting dan berpengaruh terhadap performa model. Jika memilih k yang terlalu rendah, maka akan menghasilkan model yang overfitting dan hasil prediksinya memiliki varians tinggi. Sedangkan jika memilih k yang terlalu tinggi, maka model yang dihasilkan akan underfitting dan prediksinya memiliki bias yang tinggi .

   Oleh karena itu, perlu mencoba beberapa nilai k yang berbeda (1 sampai 20) kemudian membandingan mana yang menghasilkan nilai metrik model (pada kasus ini memakai mean squared error) terbaik. Selain itu, akan digunakan metrik ukuran jarak secara default (Minkowski Distance) pada KNeighborsRegressor dari library sklearn.

   kemudian apabila ditampilkan menggunkan plot sbg berikut :
   |[<img src="/image/7.png"/>](/image/7.png)|
   |:--:| 
   | *Gambar 7. Visualisasi Nilai K terhadap MSE.* |
   Berdasarkan plot tersebut Nilai K = 2 menghasilkan MSE paling kecil

### 5.2 Random Forest
   Random forest merupakan algoritma supervised learning yang termasuk ke dalam kategori ensemble (group) learning. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir. Jenis metode ensemble yang digunakan pada Random Forest adalah teknik Bagging. Metode ini bekerja dengan membuat subset dari data train yang independen. Beberapa model awal (base model / weak model) dibuat untuk dijalankan secara simultan / paralel dan independen satu sama lain dengan subset data train yang independen. Hasil prediksi setiap model kemudian dikombinasikan untuk menentukan hasil prediksi final.

   Untuk implementasinya menggunakan RandomForestRegressor dari library scikit-learn dengan base_estimator defaultnya yaitu DecisionTreeRegressor dan parameter-parameter (hyperparameter) yang digunakan antara lain:

   n_estimator: jumlah trees (pohon) di forest.
   max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.

   | |KNN|RandomForest|
   |:----|:----|:----|
   |Train MSE|1.911107|0.513059|
   |Test MSE|3.255506|3.32484|

kemudian apabila ditampilkan menggunkan plot sbg berikut :
   |[<img src="/image/8.png"/>](/image/8.png)|
   |:--:| 
   | *Gambar 8. Bar Chart Hasil Evaluasi Model dengan Data Latih dan Uji.* |
   Dari gambar di atas, terlihat bahwa, model RandomForest memberikan nilai eror (MSE) yang paling kecil. Sedangkan model algoritma KNN memiliki eror yang paling besar dan model RandomForest akan dipertimbangkan untuk memprediksi suatu match nantinya.

## 6. Evaluation
Evaluasi kinerja pemodelan *machine learning* dilakukan dengan beberapa cara.

Pada tahapan evaluasi ini pemodelan dengan menggunakan *Random Forest* diukur kinerjanya dengan beberapa metriks. Berikut ini penjelasannya:

### 6.2.a. **Confusion Matrix**

- Hasil pembuatan *confusion matrix* dari perbandingan antara keluaran riil dari Nilai Y (team_1 score) dengan keluaran prediktif Nilai X (team_1 score) adalah sebagai berikut:
|[<img src="/image/9.png"/>](/image/9.png)|
|:--:| 
| *Gambar 9. Confusion Matrix untuk home team.* |
terlihat hasil *confusion matrix* tersebut memprediksi skor imbang lebih dominan dan juga untuk kemenangan skor margin kecil melawan away team lebih terlihat

- Hasil pembuatan *confusion matrix* dari perbandingan antara keluaran riil dari Nilai Y (team_2 score) dengan keluaran prediktif Nilai X (team_2 score) adalah sebagai berikut:

|[<img src="/image/10.png"/>](/image/10.png)|
|:--:| 
| *Gambar 10. Confusion Matrix untuk home team.* |
terlihat hasil *confusion matrix* tersebut memprediksi skor imbang lebih dominan dan juga untuk kemenangan skor margin kecil melawan home team lebih terlihat


|[<img src="/image/11.png"/>](/image/11.png)|
|:--:| 
| *Gambar 11. Memprediksi pertandingan antara USA vs Netherlands pada stadion Qatar.* |

|[<img src="/image/12.png"/>](/image/12.png)|
|:--:| 
| *Gambar 12. Hasil pertandingan antara USA vs Netherlands pada stadion Qatar.* |

## 7. Kesimpulan
   Berdasarkan algoritma, Random Forest lebih baik dibandingkan dengan KNN dalam hal memprediksi sederhana dari suatu match menggunakan record dari sebuah skor tanpa mempertimbangkan aspek lainya






![12](https://user-images.githubusercontent.com/84785750/205471514-e0a3340c-772c-4e9d-8a40-a96d6a268a9c.png)





