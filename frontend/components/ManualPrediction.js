import { useState } from 'react'

export default function ManualPrediction() {
  const [formData, setFormData] = useState({
    name: '',
    age: '',
    gender: 'Laki-laki',
    pregnancy: 'Tidak',
    polyuria: 'Tidak',
    polydipsia: 'Tidak',
    weight_loss: 'Tidak',
    weakness: 'Tidak',
    polyphagia: 'Tidak',
    genital_thrush: 'Tidak',
    visual_blurring: 'Tidak',
    itching: 'Tidak',
    irritability: 'Tidak',
    delayed_healing: 'Tidak',
    partial_paresis: 'Tidak',
    muscle_stiffness: 'Tidak',
    alopecia: 'Tidak',
    obesity: 'Tidak',
    genetics: 'Tidak'
  })
  
  const [modelType, setModelType] = useState('rf_baseline')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('http://localhost:5000/api/predict/manual', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...formData,
          model_type: modelType
        })
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed')
      }

      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const fields = [
    { name: 'name', label: 'Nama Pasien', type: 'text' },
    { name: 'age', label: 'Usia', type: 'number' },
    { name: 'gender', label: 'Jenis Kelamin', type: 'select', options: ['Laki-laki', 'Perempuan'] },
    { name: 'pregnancy', label: 'Kehamilan (Pernah/Sedang)', type: 'select', options: ['Ya', 'Tidak'] },
    { name: 'polyuria', label: 'Poliuria (Sering kencing)', type: 'select', options: ['Ya', 'Tidak'] },
    { name: 'polydipsia', label: 'Polidipsia (Sering haus)', type: 'select', options: ['Ya', 'Tidak'] },
    { name: 'weight_loss', label: 'Penurunan Berat Badan Tiba-tiba', type: 'select', options: ['Ya', 'Tidak'] },
    { name: 'weakness', label: 'Mudah Lelah (Weakness)', type: 'select', options: ['Ya', 'Tidak'] },
    { name: 'polyphagia', label: 'Polifagia (Sering lapar)', type: 'select', options: ['Ya', 'Tidak'] },
    { name: 'genital_thrush', label: 'Infeksi Jamur (Genital Thrush)', type: 'select', options: ['Ya', 'Tidak'] },
    { name: 'visual_blurring', label: 'Penglihatan Kabur', type: 'select', options: ['Ya', 'Tidak'] },
    { name: 'itching', label: 'Gatal (Itching)', type: 'select', options: ['Ya', 'Tidak'] },
    { name: 'irritability', label: 'Mudah Marah (Irritability)', type: 'select', options: ['Ya', 'Tidak'] },
    { name: 'delayed_healing', label: 'Penyembuhan Luka Lambat', type: 'select', options: ['Ya', 'Tidak'] },
    { name: 'partial_paresis', label: 'Kesemutan / Paresis', type: 'select', options: ['Ya', 'Tidak'] },
    { name: 'muscle_stiffness', label: 'Kekakuan Otot', type: 'select', options: ['Ya', 'Tidak'] },
    { name: 'alopecia', label: 'Kerontokan Rambut', type: 'select', options: ['Ya', 'Tidak'] },
    { name: 'obesity', label: 'Obesitas', type: 'select', options: ['Ya', 'Tidak'] },
    { name: 'genetics', label: 'Riwayat Genetik Diabetes', type: 'select', options: ['Ya', 'Tidak'] },
  ]

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">Prediksi Manual (Input Form)</h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700">Pilih Model</label>
          <select 
            value={modelType} 
            onChange={(e) => setModelType(e.target.value)}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
          >
            <option value="rf_baseline">Random Forest (Full Features)</option>
            <option value="lr_rfe">Logistic Regression (RFE Selected Features)</option>
          </select>
          <p className="text-xs text-gray-500 mt-1">
            {modelType === 'rf_baseline' ? 'Menggunakan semua 18 indikator.' : 'Menggunakan 8 indikator hasil seleksi RFE.'}
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {fields.map((field) => (
            <div key={field.name} className={modelType === 'lr_rfe' && !['age', 'gender', 'pregnancy', 'polydipsia', 'weight_loss', 'weakness', 'visual_blurring', 'partial_paresis'].includes(field.name) ? 'opacity-50' : ''}>
              <label className="block text-sm font-medium text-gray-700">{field.label}</label>
              {field.type === 'select' ? (
                <select
                  name={field.name}
                  value={formData[field.name]}
                  onChange={handleChange}
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                >
                  {field.options.map(opt => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              ) : (
                <input
                  type={field.type}
                  name={field.name}
                  value={formData[field.name]}
                  onChange={handleChange}
                  required
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                />
              )}
            </div>
          ))}
        </div>

        <div className="mt-6">
          <button
            type="submit"
            disabled={loading}
            className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${loading ? 'bg-indigo-400' : 'bg-indigo-600 hover:bg-indigo-700'} focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500`}
          >
            {loading ? 'Memproses...' : 'Prediksi Sekarang'}
          </button>
        </div>
      </form>

      {error && (
        <div className="mt-4 p-4 bg-red-100 text-red-700 rounded-md">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-6 p-6 bg-gray-50 rounded-lg border border-gray-200 text-center">
          <h3 className="text-lg font-medium text-gray-900">Hasil Prediksi untuk {formData.name}</h3>
          <div className="mt-4 flex flex-col items-center">
            <div className={`text-4xl font-bold ${result.prediction === 1 ? 'text-red-600' : 'text-green-600'}`}>
              {result.prediction === 1 ? 'POSITIF DIABETES' : 'NEGATIF DIABETES'}
            </div>
            {result.probability !== null && (
              <div className="mt-2 text-gray-600">
                Tingkat Keyakinan: <span className="font-semibold">{(result.probability * 100).toFixed(2)}%</span>
              </div>
            )}
            <div className="mt-2 text-sm text-gray-500">
              Model: {result.model_used === 'rf_baseline' ? 'Random Forest Baseline' : 'Logistic Regression RFE'}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}