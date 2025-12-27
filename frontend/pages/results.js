import { useState, useEffect } from 'react'
import Head from 'next/head'
import axios from 'axios'
import Link from 'next/link'
import { Bar, Radar } from 'react-chartjs-2'
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend } from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend)

const getBestModel = (results) => {
  let best = { score: -1, modelName: '', selectionMethod: '', metrics: {} }
  
  const methods = ['baseline', 'rfe', 'bfs']
  const models = ['random_forest', 'svm', 'logistic_regression']
  
  methods.forEach(method => {
    if (!results[method]) return
    models.forEach(model => {
      if (!results[method][model]) return
      const metrics = results[method][model]
      if (metrics.accuracy > best.score) {
        best = {
          score: metrics.accuracy,
          modelName: model.replace('_', ' ').toUpperCase(),
          selectionMethod: method.toUpperCase(),
          metrics: metrics
        }
      }
    })
  })
  return best
}

export default function Results() {
  const [sessions, setSessions] = useState([])
  const [selectedSession, setSelectedSession] = useState(null)
  const [loading, setLoading] = useState(true)
  const [quickResult, setQuickResult] = useState(null)

  useEffect(() => {
    fetchSessions()
    const stored = localStorage.getItem('predictionResults')
    if (stored) {
      setQuickResult(JSON.parse(stored))
    }
  }, [])

  const fetchSessions = async () => {
    try {
      const res = await axios.get('http://localhost:5000/api/training/sessions')
      setSessions(res.data)
      setLoading(false)
    } catch (err) {
      console.error(err)
      setLoading(false)
    }
  }

  const handleSelectSession = async (sid) => {
    try {
      const res = await axios.get(`http://localhost:5000/api/training/${sid}`)
      setSelectedSession(res.data)
      setQuickResult(null) // Clear quick result when selecting a session
    } catch (err) {
      console.error(err)
    }
  }
  
  const handleSelectQuickResult = () => {
    setSelectedSession(null)
    const stored = localStorage.getItem('predictionResults')
    if (stored) setQuickResult(JSON.parse(stored))
  }

  const handleDeleteSession = async (sid) => {
    if (!confirm('Apakah Anda yakin ingin menghapus sesi ini?')) return
    try {
      await axios.delete(`http://localhost:5000/api/training/${sid}`)
      fetchSessions()
      if (selectedSession && selectedSession.id === sid) {
        setSelectedSession(null)
      }
    } catch (err) {
      console.error(err)
    }
  }

  // Helper to render metrics for Quick Result (which has different structure)
  const renderQuickResult = () => {
    if (!quickResult) return null
    const bestModel = getBestModel(quickResult)
    // Reuse the visualization logic or components if possible
    // For now, simple render to ensure it works
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Hasil Quick Upload</h2>
        <div className="mb-6">
           <h3 className="text-lg font-semibold">Model Terbaik: {bestModel.modelName} ({bestModel.selectionMethod})</h3>
           <p>Akurasi: {(bestModel.metrics.accuracy * 100).toFixed(2)}%</p>
           <p>F1 Score: {(bestModel.metrics.f1 * 100).toFixed(2)}%</p>
        </div>
        <div className="grid md:grid-cols-2 gap-8">
           <div>
             <h4 className="font-medium mb-2">Perbandingan Akurasi</h4>
             <Bar 
               data={{
                 labels: ['Random Forest', 'SVM', 'Logistic Regression'],
                 datasets: [
                   { label: 'Baseline', data: [quickResult.baseline.random_forest.accuracy, quickResult.baseline.svm.accuracy, quickResult.baseline.logistic_regression.accuracy], backgroundColor: 'rgba(14, 165, 233, 0.7)' },
                   { label: 'RFE', data: [quickResult.rfe.random_forest.accuracy, quickResult.rfe.svm.accuracy, quickResult.rfe.logistic_regression.accuracy], backgroundColor: 'rgba(139, 92, 246, 0.7)' },
                   { label: 'BFS', data: [quickResult.bfs.random_forest.accuracy, quickResult.bfs.svm.accuracy, quickResult.bfs.logistic_regression.accuracy], backgroundColor: 'rgba(34, 197, 94, 0.7)' }
                 ]
               }}
             />
           </div>
        </div>
      </div>
    )
  }

  return (
    <div>
      <Head>
        <title>Analisis Hasil - Prediksi Diabetes Melitus</title>
        <meta name="description" content="Analisis hasil training model" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Analisis Hasil Training</h1>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* List Sesi */}
          <div className="lg:col-span-1 bg-white p-6 rounded-lg shadow-md h-fit">
            <h2 className="text-xl font-semibold mb-4">Daftar Sesi</h2>
            
            {quickResult && (
              <div 
                className={`p-3 border rounded cursor-pointer hover:bg-gray-50 mb-4 ${!selectedSession ? 'border-primary-500 ring-1 ring-primary-500' : 'border-gray-200'}`}
                onClick={handleSelectQuickResult}
              >
                <h3 className="font-medium text-gray-900">Quick Result (Terakhir)</h3>
                <p className="text-xs text-gray-500 mt-1">Dari menu Upload Data</p>
              </div>
            )}
            
            <div className="border-t pt-4">
            {loading ? (
              <p>Loading...</p>
            ) : sessions.length === 0 ? (
              <p className="text-gray-500">Belum ada sesi training tersimpan.</p>
            ) : (
              <div className="space-y-3">
                {sessions.map((s) => (
                  <div 
                    key={s.id} 
                    className={`p-3 border rounded cursor-pointer hover:bg-gray-50 ${selectedSession?.id === s.id ? 'border-primary-500 ring-1 ring-primary-500' : 'border-gray-200'}`}
                    onClick={() => handleSelectSession(s.id)}
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-medium text-gray-900">{s.name || 'Sesi Tanpa Nama'}</h3>
                        <p className="text-xs text-gray-500 mt-1">Status: {s.status}</p>
                      </div>
                      <button 
                        onClick={(e) => { e.stopPropagation(); handleDeleteSession(s.id); }}
                        className="text-red-500 hover:text-red-700 text-xs"
                      >
                        Hapus
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
            </div>
          </div>

          {/* Detail Sesi */}
          <div className="lg:col-span-2">
            {!selectedSession && quickResult ? (
              renderQuickResult()
            ) : selectedSession ? (
              <div className="bg-white p-6 rounded-lg shadow-md">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-2xl font-bold text-gray-900">{selectedSession.name || 'Detail Sesi'}</h2>
                  <Link href={`/training?sid=${selectedSession.id}`}>
                    <a className="text-primary-600 hover:text-primary-800 text-sm font-medium">Lanjut Training &rarr;</a>
                  </Link>
                </div>

                {selectedSession.metrics ? (
                  <div className="space-y-8">
                    {/* Metrics Cards */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                      <div className="bg-gray-50 p-4 rounded-lg text-center">
                        <p className="text-sm text-gray-500">Akurasi</p>
                        <p className="text-2xl font-bold text-gray-900">{(selectedSession.metrics.accuracy * 100).toFixed(1)}%</p>
                      </div>
                      <div className="bg-gray-50 p-4 rounded-lg text-center">
                        <p className="text-sm text-gray-500">Precision</p>
                        <p className="text-2xl font-bold text-gray-900">{(selectedSession.metrics.precision * 100).toFixed(1)}%</p>
                      </div>
                      <div className="bg-gray-50 p-4 rounded-lg text-center">
                        <p className="text-sm text-gray-500">Recall</p>
                        <p className="text-2xl font-bold text-gray-900">{(selectedSession.metrics.recall * 100).toFixed(1)}%</p>
                      </div>
                      <div className="bg-gray-50 p-4 rounded-lg text-center">
                        <p className="text-sm text-gray-500">F1 Score</p>
                        <p className="text-2xl font-bold text-gray-900">{(selectedSession.metrics.f1 * 100).toFixed(1)}%</p>
                      </div>
                    </div>

                    {/* Charts */}
                    <div className="grid md:grid-cols-2 gap-8">
                      <div>
                        <h3 className="text-lg font-medium mb-4">Metrik Performa</h3>
                        <Radar 
                          data={{
                            labels: ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'],
                            datasets: [{
                              label: selectedSession.metrics.model,
                              data: [
                                selectedSession.metrics.accuracy,
                                selectedSession.metrics.precision,
                                selectedSession.metrics.recall,
                                selectedSession.metrics.f1,
                                selectedSession.metrics.roc_auc || 0
                              ],
                              backgroundColor: 'rgba(14, 165, 233, 0.2)',
                              borderColor: 'rgba(14, 165, 233, 1)',
                              borderWidth: 2,
                            }]
                          }}
                          options={{
                            scales: { r: { beginAtZero: true, max: 1 } }
                          }}
                        />
                      </div>
                      
                      {selectedSession.metrics.confusion_matrix && (
                        <div>
                          <h3 className="text-lg font-medium mb-4">Confusion Matrix</h3>
                          <div className="flex justify-center">
                            <table className="border-collapse border border-gray-300">
                              <thead>
                                <tr>
                                  <th className="p-2 border border-gray-300 bg-gray-50"></th>
                                  <th className="p-2 border border-gray-300 bg-gray-50">Pred Neg</th>
                                  <th className="p-2 border border-gray-300 bg-gray-50">Pred Pos</th>
                                </tr>
                              </thead>
                              <tbody>
                                <tr>
                                  <th className="p-2 border border-gray-300 bg-gray-50">Act Neg</th>
                                  <td className="p-4 border border-gray-300 text-center bg-green-50">{selectedSession.metrics.confusion_matrix[0][0]}</td>
                                  <td className="p-4 border border-gray-300 text-center bg-red-50">{selectedSession.metrics.confusion_matrix[0][1]}</td>
                                </tr>
                                <tr>
                                  <th className="p-2 border border-gray-300 bg-gray-50">Act Pos</th>
                                  <td className="p-4 border border-gray-300 text-center bg-red-50">{selectedSession.metrics.confusion_matrix[1][0]}</td>
                                  <td className="p-4 border border-gray-300 text-center bg-green-50">{selectedSession.metrics.confusion_matrix[1][1]}</td>
                                </tr>
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
                    <p className="text-gray-500 mb-2">Sesi ini belum memiliki hasil evaluasi.</p>
                    <Link href={`/training?sid=${selectedSession.id}`}>
                      <a className="text-primary-600 hover:text-primary-700 font-medium">Lanjutkan ke proses Training</a>
                    </Link>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-white p-12 rounded-lg shadow-md text-center">
                <p className="text-gray-500">Pilih sesi dari daftar di sebelah kiri untuk melihat detail.</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
