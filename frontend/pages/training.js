import { useEffect, useState, useCallback } from 'react'
import Head from 'next/head'
import Link from 'next/link'
import axios from 'axios'
import { useRouter } from 'next/router'

export default function Training() {
  const [sessionId, setSessionId] = useState(null)
  const [name, setName] = useState('')
  const [file, setFile] = useState(null)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState('idle')
  const [message, setMessage] = useState('')
  const [metrics, setMetrics] = useState(null)
  const [bestModel, setBestModel] = useState(null)
  const [modelPath, setModelPath] = useState(null)
  const [loading, setLoading] = useState(false)
  const [epochs, setEpochs] = useState(100)
  const [batchSize, setBatchSize] = useState(32)
  const [learningRate, setLearningRate] = useState(0.001)
  const [logs, setLogs] = useState([])
  const [comparison, setComparison] = useState(null)
  const router = useRouter()

  const createSession = async () => {
    const res = await axios.post('http://localhost:5000/api/training/session', { name })
    const sid = res.data.session_id
    setSessionId(sid)
    setStatus('created')
    setProgress(0)
    setMessage('Sesi dibuat')
    return sid
  }

  const pollStatus = useCallback(async () => {
    if (!sessionId) return
    try {
      const res = await axios.get(`http://localhost:5000/api/training/${sessionId}/status`)
      setStatus(res.data.status)
      setProgress(res.data.progress)
    } catch (e) {}
  }, [sessionId])
  
  useEffect(() => {
    if (!sessionId) return
    const t = setInterval(pollStatus, 1500)
    return () => clearInterval(t)
  }, [sessionId, pollStatus])
  
  useEffect(() => {
    const sid = router.query.sid
    if (sid && !sessionId) {
      setSessionId(sid)
      axios.get(`http://localhost:5000/api/training/${sid}`).then((res) => {
        setName(res.data.name || '')
        setStatus(res.data.status || 'idle')
        setProgress(res.data.progress || 0)
      }).catch(() => {})
    }
  }, [router.query, sessionId])
  
  const saveName = async () => {
    if (!sessionId) {
      const sid = await createSession()
      setSessionId(sid)
      return
    }
    try {
      const res = await axios.patch(`http://localhost:5000/api/training/${sessionId}/name`, { name })
      setMessage(res.data.message)
    } catch (err) {
      setMessage(err.response?.data?.error || 'Terjadi kesalahan')
    }
  }

  const handleUpload = async (e) => {
    e.preventDefault()
    let sid = sessionId
    if (!sid) {
      sid = await createSession()
    }
    if (!file) {
      setMessage('Silakan pilih file CSV')
      return
    }
    setLoading(true)
    const formData = new FormData()
    formData.append('file', file)
    const res = await axios.post(`http://localhost:5000/api/training/${sid}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    setMessage(res.data.message)
    setLoading(false)
    pollStatus()
  }

  const handlePreprocess = async () => {
    if (!sessionId) {
      setMessage('Sesi belum dibuat')
      return
    }
    setLoading(true)
    try {
      const res = await axios.post(`http://localhost:5000/api/training/${sessionId}/preprocess`)
      setMessage(res.data.message)
    } catch (err) {
      setMessage(err.response?.data?.error || 'Terjadi kesalahan')
    } finally {
      setLoading(false)
      pollStatus()
    }
  }

  const handleSplit = async () => {
    if (!sessionId) {
      setMessage('Sesi belum dibuat')
      return
    }
    setLoading(true)
    try {
      const res = await axios.post(`http://localhost:5000/api/training/${sessionId}/split`)
      setMessage(`${res.data.message} (train: ${res.data.train_size}, test: ${res.data.test_size})`)
    } catch (err) {
      setMessage(err.response?.data?.error || 'Terjadi kesalahan')
    } finally {
      setLoading(false)
      pollStatus()
    }
  }

  const handleTrain = async () => {
    if (!sessionId) {
      setMessage('Sesi belum dibuat')
      return
    }
    setLoading(true)
    setLogs([])
    setComparison(null)
    try {
      const res = await axios.post(`http://localhost:5000/api/training/${sessionId}/train`, {
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate
      }, { timeout: 300000 })
      setMessage(res.data.message)
      setBestModel({ name: res.data.best_model, trainScore: res.data.train_score })
      setLogs(res.data.logs || [])
      setComparison(res.data.comparison || null)
    } catch (err) {
      setMessage(err.response?.data?.error || 'Terjadi kesalahan')
    } finally {
      setLoading(false)
      pollStatus()
    }
  }

  const handleEvaluate = async () => {
    if (!sessionId) {
      setMessage('Sesi belum dibuat')
      return
    }
    setLoading(true)
    try {
      const res = await axios.post(`http://localhost:5000/api/training/${sessionId}/evaluate`)
      setMessage(res.data.message)
      setMetrics(res.data.metrics)
    } catch (err) {
      setMessage(err.response?.data?.error || 'Terjadi kesalahan')
    } finally {
      setLoading(false)
      pollStatus()
    }
  }

  const handleSave = async () => {
    if (!sessionId) {
      setMessage('Sesi belum dibuat')
      return
    }
    setLoading(true)
    try {
      const res = await axios.post(`http://localhost:5000/api/training/${sessionId}/save`)
      setMessage(res.data.message)
      setModelPath(res.data.model_path)
    } catch (err) {
      setMessage(err.response?.data?.error || 'Terjadi kesalahan')
    } finally {
      setLoading(false)
      pollStatus()
    }
  }

  return (
    <div>
      <Head>
        <title>Training Model - Prediksi Diabetes Melitus</title>
        <meta name="description" content="Halaman pelatihan model dengan progress bertahap" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="flex-1 max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <div className="bg-white p-6 rounded-lg shadow-md mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-700">Status: {status}</span>
              <span className="text-sm text-gray-700">Sesi: {sessionId || '-'}</span>
            </div>
            <button onClick={pollStatus} className="px-3 py-1 bg-gray-100 rounded hover:bg-gray-200 text-sm">Refresh Status</button>
          </div>
          <div className="flex items-center space-x-2 mb-3">
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Nama sesi (mis. Training 1)" className="border rounded px-3 py-1 text-sm" />
            <button onClick={saveName} className="px-3 py-1 bg-primary-600 text-white rounded text-sm">Simpan Nama</button>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5 mb-3">
            <div className="bg-primary-600 h-2.5 rounded-full" style={{ width: `${progress}%` }}></div>
          </div>
          {message && <p className="text-gray-700">{message}</p>}
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">1. Upload Data</h2>
            <form onSubmit={handleUpload} className="space-y-3">
              <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} className="block w-full" />
              <button disabled={loading} className="px-4 py-2 rounded bg-primary-600 text-white hover:bg-primary-700">
                {loading ? 'Memproses...' : 'Upload'}
              </button>
            </form>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">2. Preprocessing</h2>
            <button disabled={loading || status !== 'uploaded'} onClick={handlePreprocess} className="px-4 py-2 rounded bg-secondary-600 text-white hover:bg-secondary-700 disabled:opacity-50">
              Mulai Preprocessing
            </button>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">3. Split Data</h2>
            <button disabled={loading || status !== 'preprocessed'} onClick={handleSplit} className="px-4 py-2 rounded bg-green-600 text-white hover:bg-green-700 disabled:opacity-50">
              Split Train/Test
            </button>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">4. Training</h2>
            
            <div className="mb-4 p-4 bg-gray-50 rounded text-sm space-y-3">
              <h3 className="font-medium text-gray-900">Konfigurasi Training</h3>
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Epochs / Max Iter</label>
                  <input 
                    type="number" 
                    value={epochs} 
                    onChange={(e) => setEpochs(e.target.value)} 
                    className="w-full border rounded px-2 py-1"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Batch Size (MLP)</label>
                  <input 
                    type="number" 
                    value={batchSize} 
                    onChange={(e) => setBatchSize(e.target.value)} 
                    className="w-full border rounded px-2 py-1"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Learning Rate</label>
                  <input 
                    type="number" 
                    step="0.0001"
                    value={learningRate} 
                    onChange={(e) => setLearningRate(e.target.value)} 
                    className="w-full border rounded px-2 py-1"
                  />
                </div>
              </div>
            </div>

            <button disabled={loading || status !== 'split'} onClick={handleTrain} className="px-4 py-2 rounded bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50 w-full mb-4">
              Mulai Training
            </button>
            
            {logs.length > 0 && (
              <div className="mt-4 p-3 bg-gray-800 text-green-400 font-mono text-xs rounded h-40 overflow-y-auto">
                {logs.map((log, i) => (
                  <div key={i}>{log}</div>
                ))}
              </div>
            )}

            {comparison && (
              <div className="mt-4 overflow-x-auto">
                <h3 className="font-medium text-gray-900 mb-2">Perbandingan Model</h3>
                <table className="min-w-full text-xs text-left text-gray-500">
                    <thead className="bg-gray-50 text-gray-700 uppercase">
                        <tr>
                            <th className="px-3 py-2">Phase</th>
                            <th className="px-3 py-2">Model</th>
                            <th className="px-3 py-2">Accuracy</th>
                            <th className="px-3 py-2">Time (s)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {comparison.map((row, idx) => (
                            <tr key={idx} className="border-b">
                                <td className="px-3 py-2 font-medium text-gray-900">{row.phase}</td>
                                <td className="px-3 py-2">{row.model}</td>
                                <td className="px-3 py-2">{(row.accuracy * 100).toFixed(2)}%</td>
                                <td className="px-3 py-2">{row.duration.toFixed(3)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
              </div>
            )}

            {bestModel && (
              <div className="mt-4 p-3 bg-indigo-50 border border-indigo-100 rounded">
                <p className="font-medium text-indigo-900">Hasil Terbaik:</p>
                <p className="text-indigo-800">{bestModel.name}</p>
                <p className="text-sm text-indigo-600">Train Score: {(bestModel.trainScore * 100).toFixed(2)}%</p>
              </div>
            )}
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">5. Evaluasi</h2>
            <button disabled={loading || status !== 'trained'} onClick={handleEvaluate} className="px-4 py-2 rounded bg-yellow-500 text-white hover:bg-yellow-600 disabled:opacity-50">
              Evaluasi
            </button>
            {metrics && (
              <div className="mt-3 text-gray-700">
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
            )}
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">6. Simpan Model</h2>
            <button disabled={loading || status !== 'evaluated'} onClick={handleSave} className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50">
              Simpan
            </button>
            {modelPath && (
              <p className="mt-3 text-gray-700 break-all">Lokasi model: {modelPath}</p>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
