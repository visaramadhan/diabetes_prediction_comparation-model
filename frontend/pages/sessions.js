import { useEffect, useState } from 'react'
import Head from 'next/head'
import Link from 'next/link'
import axios from 'axios'

export default function Sessions() {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [newName, setNewName] = useState('')

  useEffect(() => {
    axios.get('http://localhost:5000/api/training/sessions')
      .then(res => setSessions(res.data))
      .catch(err => setError(err.response?.data?.error || 'Terjadi kesalahan'))
      .finally(() => setLoading(false))
  }, [])

  const refresh = () => {
    setLoading(true)
    axios.get('http://localhost:5000/api/training/sessions')
      .then(res => setSessions(res.data))
      .catch(err => setError(err.response?.data?.error || 'Terjadi kesalahan'))
      .finally(() => setLoading(false))
  }

  const createSession = async () => {
    try {
      await axios.post('http://localhost:5000/api/training/session', { name: newName })
      setNewName('')
      refresh()
    } catch (err) {
      setError(err.response?.data?.error || 'Gagal membuat sesi')
    }
  }

  const deleteSession = async (id) => {
    try {
      await axios.delete(`http://localhost:5000/api/training/${id}`)
      refresh()
    } catch (err) {
      setError(err.response?.data?.error || 'Gagal menghapus sesi')
    }
  }

  const renameSession = async (id, name) => {
    try {
      await axios.patch(`http://localhost:5000/api/training/${id}/name`, { name })
      refresh()
    } catch (err) {
      setError(err.response?.data?.error || 'Gagal mengganti nama sesi')
    }
  }

  return (
    <div>
      <Head>
        <title>Daftar Sesi - Prediksi Diabetes Melitus</title>
        <meta name="description" content="Daftar sesi training yang tersimpan" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="max-w-6xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">Daftar Sesi</h1>
        <div className="flex items-center space-x-2 mb-6">
          <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Nama sesi baru" className="border rounded px-3 py-2 text-sm" />
          <button onClick={createSession} className="px-4 py-2 bg-primary-600 text-white rounded text-sm">Buat Sesi</button>
          <button onClick={refresh} className="px-4 py-2 bg-gray-100 rounded text-sm">Refresh</button>
        </div>
        {loading && <p className="text-gray-600">Memuat...</p>}
        {error && <p className="text-red-600">{error}</p>}
        {!loading && !error && (
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <table className="min-w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Nama</th>
                  <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Status</th>
                  <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Progress</th>
                  <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Aksi</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {sessions.map(s => (
                  <tr key={s.id}>
                    <td className="px-4 py-2">
                      <input defaultValue={s.name || ''} className="border rounded px-2 py-1 text-sm" onBlur={(e) => renameSession(s.id, e.target.value)} />
                    </td>
                    <td className="px-4 py-2">{s.status}</td>
                    <td className="px-4 py-2">{s.progress}%</td>
                    <td className="px-4 py-2">
                      <Link href={`/training?sid=${s.id}`}>
                        <a className="text-primary-600 hover:text-primary-700">Buka</a>
                      </Link>
                      <button onClick={() => deleteSession(s.id)} className="ml-3 text-red-600 hover:text-red-700">Hapus</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </div>
  )
}
