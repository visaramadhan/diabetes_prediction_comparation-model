import { useState } from 'react'
import Head from 'next/head'
import { useRouter } from 'next/router'
import axios from 'axios'
import Link from 'next/link'

export default function Upload() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [headerValid, setHeaderValid] = useState(true)
  const requiredHeaders = ['Usia','Jenis Kelamin','Riwayat Keluarga','BMI','Tekanan Darah','Gula Darah','Kehamilan','Kebiasaan Merokok','Aktivitas Fisik','Pola Tidur','Diagnosis']
  const router = useRouter()

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    setFile(selectedFile)
    setError(null)
    setHeaderValid(true)
    if (selectedFile) {
      const reader = new FileReader()
      reader.onload = (evt) => {
        const text = evt.target.result
        const firstLine = typeof text === 'string' ? text.split(/\r?\n/)[0] : ''
        const headers = firstLine.split(',').map(h => h.trim())
        const normalize = (s) => s.toLowerCase().trim()
        const requiredNorm = requiredHeaders.map(normalize)
        const headerNorm = headers.map(normalize)
        const missing = requiredNorm
          .filter(h => !headerNorm.includes(h))
          .map(h => {
            const i = requiredNorm.indexOf(h)
            return i >= 0 ? requiredHeaders[i] : h
          })
        if (missing.length > 0) {
          setHeaderValid(false)
          setError(`Header CSV tidak sesuai. Kolom yang hilang: ${missing.join(', ')}`)
        }
      }
      reader.readAsText(selectedFile)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!file) {
      setError('Silakan pilih file terlebih dahulu')
      return
    }
    
    const name = (file.name || '').toLowerCase()
    const isCsvExt = name.endsWith('.csv')
    const isCsvMime = file.type === 'text/csv' || file.type === 'application/vnd.ms-excel'
    if (!isCsvExt && !isCsvMime) {
      setError('File harus berformat CSV')
      return
    }
    
    if (!headerValid) {
      setError('Perbaiki header CSV sesuai format yang diwajibkan')
      return
    }
    
    setLoading(true)
    setError(null)
    
    const formData = new FormData()
    formData.append('file', file)
    
    try {
      const response = await axios.post('http://localhost:5000/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      
      // Simpan hasil ke localStorage untuk digunakan di halaman results
      localStorage.setItem('predictionResults', JSON.stringify(response.data))
      
      // Redirect ke halaman results
      router.push('/results')
    } catch (error) {
      console.error('Error uploading file:', error)
      setError(error.response?.data?.error || 'Terjadi kesalahan saat mengunggah file')
      setLoading(false)
    }
  }

  return (
    <div>
      <Head>
        <title>Upload Data - Prediksi Diabetes Melitus</title>
        <meta name="description" content="Upload data untuk prediksi diabetes melitus" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="py-12 px-4 sm:px-6 lg:px-8">
        <div className="w-full max-w-3xl"></div>
          <h1 className="text-3xl font-bold text-gray-900 text-center mb-8">Upload Data</h1>
          
          <div className="bg-white shadow-md rounded-lg overflow-hidden">
            <form onSubmit={handleSubmit} className="p-6 border-b border-gray-200">
              <div className="mb-6">
                <label htmlFor="file" className="block text-sm font-medium text-gray-700 mb-2">Pilih file CSV:</label>
                <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md hover:border-primary-300 transition-colors">
                  <div className="space-y-1 text-center">
                    <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                      <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    <div className="flex text-sm text-gray-600">
                      <label htmlFor="file" className="relative cursor-pointer bg-white rounded-md font-medium text-primary-600 hover:text-primary-500 focus-within:outline-none">
                        <span>Upload file</span>
                        <input 
                          type="file" 
                          id="file" 
                          name="file"
                          accept=".csv" 
                          onChange={handleFileChange} 
                          disabled={loading}
                          className="sr-only"
                        />
                      </label>
                      <p className="pl-1">atau drag and drop</p>
                    </div>
                    <p className="text-xs text-gray-500">{file ? file.name : 'CSV hingga 10MB'}</p>
                  </div>
                </div>
              </div>
              
              {error && <div className="p-4 mb-4 text-sm text-red-700 bg-red-100 rounded-lg" role="alert">{error}</div>}
              
              <div className="flex justify-center">
                <button 
                  type="submit" 
                  className={`px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 ${loading || !file ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={loading || !file}
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Memproses...
                    </>
                  ) : 'Upload dan Proses'}
                </button>
              </div>
            </form>
            
            <div className="bg-gray-50 p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Format File CSV</h2>
              <p className="text-sm text-gray-600 mb-3">File CSV harus memiliki header dan berisi kolom-kolom berikut:</p>
              <div className="bg-gray-100 p-3 rounded-md overflow-x-auto mb-4">
                <code className="text-xs text-gray-800">Usia,Jenis Kelamin,Riwayat Keluarga,BMI,Tekanan Darah,Gula Darah,Kehamilan,Kebiasaan Merokok,Aktivitas Fisik,Pola Tidur,Diagnosis</code>
              </div>
              
              <p className="text-sm text-gray-600 mb-2">Dimana:</p>
              <ul className="text-sm text-gray-600 space-y-1 list-disc pl-5">
                <li><span className="font-medium">Usia</span>: Usia pasien (numerik)</li>
                <li><span className="font-medium">Jenis Kelamin</span>: 1 laki-laki, 0 perempuan</li>
                <li><span className="font-medium">Riwayat Keluarga</span>: 1 jika ada, 0 jika tidak</li>
                <li><span className="font-medium">BMI</span>: Indeks massa tubuh (numerik)</li>
                <li><span className="font-medium">Tekanan Darah</span>: nilai tekanan darah (numerik)</li>
                <li><span className="font-medium">Gula Darah</span>: kadar gula darah (numerik)</li>
                <li><span className="font-medium">Kehamilan</span>: jumlah kehamilan atau 1/0 (sesuai dataset)</li>
                <li><span className="font-medium">Kebiasaan Merokok</span>: 1 jika merokok, 0 jika tidak</li>
                <li><span className="font-medium">Aktivitas Fisik</span>: tingkat aktivitas (numerik atau kategori dibinerkan)</li>
                <li><span className="font-medium">Pola Tidur</span>: durasi/kualitas (numerik atau kategori dibinerkan)</li>
                <li><span className="font-medium">Diagnosis</span>: 1 positif DM, 0 negatif</li>
              </ul>
          </div>
        </div>
      </main>
    </div>
  )
}
