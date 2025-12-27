# BAB 4
# IMPLEMENTASI DAN PENGUJIAN SISTEM

## 4.1 Implementasi Sistem
Sistem prediksi diabetes mellitus ini dikembangkan berbasis web menggunakan framework Next.js untuk sisi antarmuka (frontend) dan Flask (Python) untuk sisi server (backend). Implementasi sistem mencakup penanganan input data, pemrosesan awal data (preprocessing), pelatihan model machine learning, serta visualisasi hasil prediksi.

## 4.2 Data Preparation
Pada tahap ini, data mentah yang diperoleh dari sumber awal terlebih dahulu diperiksa secara menyeluruh untuk mengidentifikasi permasalahan kualitas data, seperti nilai hilang, inkonsistensi format, dan potensi duplikasi data. Nilai hilang yang ditemukan pada beberapa atribut ditangani menggunakan teknik imputasi yang sesuai dengan karakteristik data, sehingga tidak mengurangi representativitas informasi yang terkandung di dalam dataset.

### 4.2.1 Data Cleansing
Pada tahap ini, peneliti melakukan pemeriksaan awal terhadap struktur data guna memastikan bahwa format dan konten data telah sesuai dengan kebutuhan penelitian. Proses ini mencakup validasi format serta konsistensi data, termasuk verifikasi bahwa atribut tertentu, seperti jenis kelamin, hanya memuat nilai yang valid. Sistem secara otomatis menormalisasi nama kolom untuk memastikan konsistensi (misalnya mengubah 'gender' atau 'jenis kelamin' menjadi format standar).

### 4.2.2 Missing Value
Pada tahap ini dilakukan proses peninjauan dan perbaikan data terhadap atribut-atribut yang berpotensi tidak terdeskripsikan. Sistem menggunakan strategi imputasi otomatis di mana nilai numerik yang hilang diisi dengan median dari kolom tersebut, sedangkan nilai kategorikal diisi dengan modus (nilai yang paling sering muncul). Baris data yang tidak memiliki label target ('Diagnosis') dihapus untuk menjaga validitas pelatihan model.

### 4.2.3 Konversi Data Kategorikal
Sebelum data dapat digunakan untuk pelatihan model, atribut kategorikal (seperti 'Jenis Kelamin', 'Riwayat Keluarga') dikonversi menjadi format numerik. Sistem menerapkan teknik *One-Hot Encoding* untuk mengubah variabel kategorikal menjadi representasi biner yang dapat diproses oleh algoritma machine learning. Hal ini memastikan bahwa tidak ada asumsi urutan yang salah pada data nominal.

### 4.2.4 Penentuan Data Latih dan Data Uji
Untuk menguji performa model secara objektif, dataset dibagi menjadi dua bagian: data latih (*training set*) dan data uji (*testing set*). Pembagian dilakukan dengan proporsi 80:20, di mana 80% data digunakan untuk melatih model dan 20% sisanya disimpan sebagai data uji yang tidak pernah dilihat oleh model selama proses pelatihan. Teknik *stratified sampling* digunakan untuk memastikan bahwa proporsi kelas target (Positif/Negatif Diabetes) seimbang antara data latih dan data uji, sehingga mencegah bias pada evaluasi model.

## 4.3 Modelling
Tahap pemodelan merupakan inti dari sistem ini, di mana berbagai algoritma pembelajaran mesin diterapkan untuk mempelajari pola dari data latih. Sistem ini menerapkan pendekatan komparatif dengan menguji empat algoritma utama: Random Forest, Support Vector Machine (SVM), Logistic Regression, dan Neural Network (Multi-Layer Perceptron).

### 4.3.1 Penerapan Model dengan Seluruh Fitur (Baseline)
Sebagai langkah awal, keempat algoritma dilatih menggunakan seluruh fitur yang tersedia dalam dataset tanpa melakukan seleksi fitur. Tujuannya adalah untuk mendapatkan *baseline performance* atau tolak ukur awal kinerja model. Pada tahap ini, semua atribut seperti usia, BMI, tekanan darah, gula darah, dan lainnya digunakan sebagai input. Hasil akurasi dan waktu komputasi dari setiap model dicatat untuk dibandingkan dengan tahap selanjutnya.

### 4.3.2 Penerapan Seleksi Fitur
Untuk meningkatkan efisiensi dan akurasi model, sistem menerapkan dua teknik seleksi fitur: *Recursive Feature Elimination* (RFE) dan *Boruta*. RFE bekerja dengan secara iteratif menghapus fitur yang paling tidak penting berdasarkan bobot yang dihasilkan oleh estimator, hingga tersisa subset fitur yang optimal. Sementara itu, Boruta bekerja sebagai *wrapper* di sekitar algoritma Random Forest, yang membandingkan pentingnya fitur asli dengan fitur bayangan (*shadow features*) yang diacak, untuk menentukan fitur mana yang benar-benar relevan secara statistik.

### 4.3.3 Penerapan Model Setelah Seleksi Fitur
Setelah subset fitur terbaik diidentifikasi oleh RFE dan Boruta, keempat model (Random Forest, SVM, Logistic Regression, Neural Network) dilatih kembali menggunakan hanya fitur-fitur terpilih tersebut. Proses ini dilakukan dalam dua iterasi terpisah: satu menggunakan fitur hasil seleksi RFE, dan satu lagi menggunakan fitur hasil seleksi Boruta. Pengurangan dimensi data ini diharapkan dapat mengurangi *overfitting*, mempercepat waktu komputasi, dan berpotensi meningkatkan akurasi prediksi dengan menghilangkan *noise* dari fitur yang tidak relevan.

### 4.3.4 Evaluasi Performa
Evaluasi dilakukan dengan menguji model-model yang telah dilatih menggunakan data uji (*testing set*) yang telah dipisahkan sebelumnya. Metrik evaluasi yang digunakan meliputi Akurasi (*Accuracy*), Presisi (*Precision*), *Recall*, dan *F1-Score*. Selain itu, matriks kebingungan (*Confusion Matrix*) dihasilkan untuk memvisualisasikan jumlah prediksi benar positif, benar negatif, salah positif, dan salah negatif. Hal ini memberikan gambaran yang lebih komprehensif mengenai kinerja model, terutama dalam meminimalisir kesalahan diagnosis palsu.

### 4.3.5 Analisis Komparatif dan Interpretasi
Berdasarkan hasil pengujian dari tiga skenario (Baseline, RFE, dan Boruta), dilakukan analisis perbandingan untuk menentukan kombinasi model dan teknik seleksi fitur terbaik. Analisis ini tidak hanya melihat skor akurasi tertinggi, tetapi juga mempertimbangkan kompleksitas model dan waktu eksekusi. Hasil eksperimen ditampilkan dalam bentuk tabel perbandingan yang memudahkan pengguna untuk melihat dampak seleksi fitur terhadap kinerja setiap algoritma. Model terbaik kemudian dipilih secara otomatis oleh sistem untuk disimpan dan digunakan dalam proses prediksi data baru.

## 4.4 Evaluation
Tahap evaluasi akhir memvalidasi model terbaik yang terpilih pada data uji independen. Hasil evaluasi menunjukkan metrik performa final yang akan menjadi acuan keandalan sistem saat digunakan di lingkungan produksi. Visualisasi berupa kurva ROC (*Receiver Operating Characteristic*) dan nilai AUC (*Area Under Curve*) juga dapat dipertimbangkan untuk mengukur kemampuan model dalam membedakan kelas positif dan negatif pada berbagai ambang batas klasifikasi.

## 4.5 Deployment
Model terbaik yang telah dilatih dan dievaluasi kemudian disimpan (*serialized*) menggunakan format `joblib` bersama dengan pipeline preprocessing-nya. Sistem menyediakan antarmuka API dan halaman web interaktif yang memungkinkan pengguna untuk mengunggah data baru dalam format CSV atau memasukkan nilai parameter secara manual. Saat data baru dimasukkan, sistem secara otomatis memuat model tersimpan, melakukan preprocessing yang sama, dan menghasilkan prediksi diagnosis diabetes secara *real-time*.

## 4.6 Prototype
Prototipe sistem dibangun dengan antarmuka yang responsif dan mudah digunakan (*user-friendly*). Halaman utama menampilkan dasbor pelatihan di mana pengguna dapat memantau progres preprocessing, splitting, dan training. Fitur 'Live Prediction' memungkinkan pengujian cepat model, sementara halaman 'Results' menyimpan riwayat analisis yang telah dilakukan. Navigasi sisi (*sidebar*) memudahkan akses ke seluruh fitur sistem, memastikan pengalaman pengguna yang efisien dalam melakukan analisis data medis.
