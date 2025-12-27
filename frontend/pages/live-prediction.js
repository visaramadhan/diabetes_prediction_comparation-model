import { useState, useEffect } from 'react'
import Head from 'next/head'
import Link from 'next/link'
import axios from 'axios'

export default function LivePrediction() {
  const [file, setFile] = useState(null)
  const [modelId, setModelId] = useState('')
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(false)
  const [predictions, setPredictions] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    // Fetch available sessions for dropdown
    axios.get('http://localhost:5000/api/training/sessions')
      .then(res => {
        // Filter only completed sessions that have models
        const completed = res.data.filter(s => s.status === 'completed')
        setSessions(completed)
        if (completed.length > 0) {
          setModelId(completed[0].id)
        }
      })
      .catch(console.error)
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) {
      setError('Silakan pilih file CSV')
      return
    }
    setLoading(true)
    setError(null)
    setPredictions(null)
    setMetrics(null)
    const formData = new FormData()
    formData.append('file', file)
    if (modelId) formData.append('model_id', modelId)
    try {
      const res = await axios.post('http://localhost:5000/api/predict/live', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setPredictions(res.data.predictions)
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
            <div className="space-y-2 text-gray-700">
              <p>Total baris: {predictions.length}</p>
              <p>Contoh 10 prediksi pertama:</p>
              <pre className="p-3 bg-gray-100 rounded">{JSON.stringify(predictions.slice(0, 10), null, 2)}</pre>
            </div>
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
      </main>
    </div>
  )
}
