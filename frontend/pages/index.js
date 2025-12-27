import Head from 'next/head'
import Link from 'next/link'

export default function Home() {
  return (
    <div>
      <Head>
        <title>Prediksi Diabetes Melitus</title>
        <meta name="description" content="Aplikasi prediksi diabetes melitus dengan komparasi 3 model" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="max-w-7xl mx-auto py-8 px-4">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Prediksi Diabetes Melitus</h1>

        <p className="text-gray-600 mb-8">Aplikasi prediksi diabetes melitus dengan komparasi 3 model machine learning</p>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          <Link href="/upload">
            <a className="block p-6 rounded-lg border bg-white hover:border-primary-300 hover:shadow-sm">
              <h2 className="text-lg font-semibold text-gray-900">Upload Data</h2>
              <p className="text-sm text-gray-600 mt-2">Upload file CSV untuk prediksi diabetes melitus.</p>
            </a>
          </Link>

          <Link href="/training">
            <a className="block p-6 rounded-lg border bg-white hover:border-primary-300 hover:shadow-sm">
              <h2 className="text-lg font-semibold text-gray-900">Training Model</h2>
              <p className="text-sm text-gray-600 mt-2">Latih model secara bertahap dengan progress di layar.</p>
            </a>
          </Link>

          <Link href="/live-prediction">
            <a className="block p-6 rounded-lg border bg-white hover:border-primary-300 hover:shadow-sm">
              <h2 className="text-lg font-semibold text-gray-900">Live Prediction</h2>
              <p className="text-sm text-gray-600 mt-2">Gunakan model tersimpan untuk prediksi data baru.</p>
            </a>
          </Link>

          <Link href="/about">
            <a className="block p-6 rounded-lg border bg-white hover:border-primary-300 hover:shadow-sm">
              <h2 className="text-lg font-semibold text-gray-900">Tentang Aplikasi</h2>
              <p className="text-sm text-gray-600 mt-2">Informasi tentang aplikasi dan model yang digunakan.</p>
            </a>
          </Link>
        </div>
      </main>
    </div>
  )
}
