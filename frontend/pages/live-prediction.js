import { useState, useEffect } from 'react'
import Head from 'next/head'
import Link from 'next/link'
import axios from 'axios'
import ManualPrediction from '../components/ManualPrediction'

export default function LivePrediction() {
  const [activeTab, setActiveTab] = useState('manual')
  const [file, setFile] = useState(null)
  const [modelId, setModelId] = useState('')
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(false)
  const [predictions, setPredictions] = useState(null)
  const [probabilities, setProbabilities] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState(null)

  const [modelFile, setModelFile] = useState(null)
  const [uploadLoading, setUploadLoading] = useState(false)

  const fetchSessions = () => {
    axios.get('http://localhost:5000/api/training/sessions')
      .then(res => {
        // Filter sessions that have a saved model
        const completed = res.data.filter(s => s.status === 'saved')
        setSessions(completed)
        if (completed.length > 0 && !modelId) {
          setModelId(completed[0].id)
        }
      })
      .catch(console.error)
  }

  useEffect(() => {
    fetchSessions()
  }, [])

  const handleModelUpload = async () => {
    if (!modelFile) return
    setUploadLoading(true)
    const formData = new FormData()
    formData.append('file', modelFile)
    try {
      await axios.post('http://localhost:5000/api/models/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setModelFile(null)
      alert('Model berhasil diupload!')
      fetchSessions()
    } catch (err) {
      alert('Gagal upload model: ' + (err.response?.data?.error || err.message))
    } finally {
      setUploadLoading(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) {
      setError('Silakan pilih file CSV')
      return
    }
    setLoading(true)
    setError(null)
    setPredictions(null)
    setProbabilities(null)
    setMetrics(null)
    const formData = new FormData()
    formData.append('file', file)
    if (modelId) formData.append('model_id', modelId)
    try {
      const res = await axios.post('http://localhost:5000/api/predict/live', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setPredictions(res.data.predictions)
      setProbabilities(res.data.probabilities || null)
      setMetrics(res.data.metrics || null)
    } catch (err) {
      setError(err.response?.data?.error || 'Terjadi kesalahan')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <Head>
        <title>Live Prediction - Prediksi Diabetes Melitus</title>
        <meta name="description" content="Halaman prediksi live menggunakan model tersimpan" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="flex-1 max-w-4xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        
        {/* Tab Navigation */}
        <div className="flex border-b border-gray-200 mb-6">
          <button
            className={`py-2 px-4 font-medium text-sm focus:outline-none ${activeTab === 'manual' ? 'border-b-2 border-indigo-500 text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
            onClick={() => setActiveTab('manual')}
          >
            Input Manual
          </button>
          <button
            className={`py-2 px-4 font-medium text-sm focus:outline-none ${activeTab === 'csv' ? 'border-b-2 border-indigo-500 text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
            onClick={() => setActiveTab('csv')}
          >
            Upload CSV (Batch)
          </button>
        </div>

        {activeTab === 'manual' ? (
          <ManualPrediction />
        ) : (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-lg shadow-md">
              <h2 className="text-lg font-semibold mb-4">Upload Data & Pilih Model</h2>
              <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm text-gray-700 mb-1">Pilih Model (Sesi)</label>
              <select 
                value={modelId} 
                onChange={(e) => setModelId(e.target.value)} 
                className="block w-full border rounded px-3 py-2"
              >
                <option value="">-- Pilih Model Tersimpan --</option>
                {sessions.map(s => (
                  <option key={s.id} value={s.id}>{s.name || s.id} (Acc: {(s.metrics?.accuracy * 100)?.toFixed(1)}%)</option>
                ))}
              </select>
              {sessions.length === 0 && <p className="text-xs text-orange-500 mt-1">Belum ada model yang selesai dilatih. Silakan lakukan Training Model terlebih dahulu.</p>}
            </div>

            <div className="border-t pt-4">
              <label className="block text-sm text-gray-700 mb-1">Atau Upload Model Eksternal (.joblib)</label>
              <div className="flex space-x-2">
                <input 
                  type="file" 
                  accept=".joblib" 
                  onChange={(e) => setModelFile(e.target.files[0])} 
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100" 
                />
                <button 
                  type="button"
                  onClick={handleModelUpload} 
                  disabled={!modelFile || uploadLoading}
                  className="px-4 py-2 bg-green-600 text-white rounded text-sm disabled:bg-gray-400"
                >
                  {uploadLoading ? 'Uploading...' : 'Upload'}
                </button>
              </div>
            </div>

            <div>
              <label className="block text-sm text-gray-700 mb-1">File CSV Data Baru</label>
              <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} className="block w-full" />
            </div>
            <button disabled={loading || !file} className={`px-4 py-2 rounded text-white ${loading || !file ? 'bg-gray-400' : 'bg-primary-600 hover:bg-primary-700'}`}>
              {loading ? 'Memproses...' : 'Prediksi'}
            </button>
          </form>
          {error && <p className="mt-3 text-red-600">{error}</p>}
        </div>

        {predictions && (
          <div className="bg-white p-6 rounded-lg shadow-md mt-6">
            <h2 className="text-lg font-semibold mb-4">Hasil Prediksi</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      No
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Prediksi
                    </th>
                    {probabilities && (
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Keyakinan (Probabilitas)
                      </th>
                    )}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {predictions.map((pred, idx) => (
                    <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {idx + 1}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                          pred === 1 || pred === 'Positif' || pred === 'Yes' 
                            ? 'bg-red-100 text-red-800' 
                            : 'bg-green-100 text-green-800'
                        }`}>
                          {pred === 1 ? 'Positif Diabetes' : pred === 0 ? 'Negatif' : pred}
                        </span>
                      </td>
                      {probabilities && (
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <div className="flex items-center">
                            <span className="mr-2">{(probabilities[idx] * 100).toFixed(2)}%</span>
                            <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                              <div 
                                className={`h-full ${probabilities[idx] > 0.5 ? 'bg-red-500' : 'bg-green-500'}`} 
                                style={{ width: `${probabilities[idx] * 100}%` }}
                              ></div>
                            </div>
                          </div>
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {predictions.length > 20 && (
              <p className="text-xs text-gray-500 mt-2 text-center">Menampilkan 20 data pertama dari {predictions.length} total baris.</p>
            )}
          </div>
        )}

        {metrics && (
          <div className="bg-white p-6 rounded-lg shadow-md mt-6">
            <h2 className="text-lg font-semibold mb-4">Metrik (jika data berlabel)</h2>
            <div className="space-y-1 text-gray-700">
              <p>Akurasi: {(metrics.accuracy * 100).toFixed(2)}%</p>
              <p>Precision: {(metrics.precision * 100).toFixed(2)}%</p>
              <p>Recall: {(metrics.recall * 100).toFixed(2)}%</p>
              <p>F1: {(metrics.f1 * 100).toFixed(2)}%</p>
              {metrics.roc_auc !== undefined && <p>ROC AUC: {(metrics.roc_auc * 100).toFixed(2)}%</p>}
              {metrics.confusion_matrix && Array.isArray(metrics.confusion_matrix) && metrics.confusion_matrix.length === 2 && (
                <div className="mt-3">
                  <p className="font-medium mb-2">Confusion Matrix</p>
                  <div className="inline-block border rounded">
                    <table className="table-fixed">
                      <tbody>
                        <tr>
                          <td className="px-3 py-2 border">{metrics.confusion_matrix[0][0]}</td>
                          <td className="px-3 py-2 border">{metrics.confusion_matrix[0][1]}</td>
                        </tr>
                        <tr>
                          <td className="px-3 py-2 border">{metrics.confusion_matrix[1][0]}</td>
                          <td className="px-3 py-2 border">{metrics.confusion_matrix[1][1]}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    )}
  </main>
    </div>
  )
}
