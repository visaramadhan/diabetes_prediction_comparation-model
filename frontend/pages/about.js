import Head from 'next/head'

export default function About() {
  return (
    <div>
      <Head>
        <title>Tentang Aplikasi - Prediksi Diabetes Melitus</title>
        <meta name="description" content="Informasi mengenai aplikasi prediksi diabetes melitus" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="max-w-4xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">Tentang Aplikasi</h1>
        <p className="text-gray-700 mb-4">
          Aplikasi ini membantu melakukan prediksi Diabetes Melitus menggunakan beberapa model pembelajaran mesin
          dan dua metode seleksi fitur. Hasil evaluasi ditampilkan agar memudahkan membandingkan performa model.
        </p>
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-3">Fitur Utama</h2>
          <ul className="list-disc pl-5 text-gray-700 space-y-2">
            <li>Upload data CSV dan proses otomatis untuk membandingkan model</li>
            <li>Alur pelatihan bertahap: upload, preprocessing, split, training, evaluasi, simpan model</li>
            <li>Prediksi live menggunakan model tersimpan dengan dukungan metrik jika data berlabel</li>
            <li>Visualisasi hasil berupa perbandingan akurasi dan radar metrik</li>
          </ul>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-3">Model dan Seleksi Fitur</h2>
          <p className="text-gray-700">
            Model yang digunakan meliputi Random Forest, Support Vector Machine, dan Logistic Regression.
            Seleksi fitur menggunakan Recursive Feature Elimination (RFE) dan Boruta.
          </p>
        </div>
      </main>
    </div>
  )
}
