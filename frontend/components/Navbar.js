import Link from 'next/link'

export default function Navbar() {
  return (
    <div className="h-14 border-b bg-white">
      <div className="h-full max-w-7xl mx-auto px-4 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-primary-600 font-semibold">Prediksi DM</span>
        </div>
        <div className="flex items-center space-x-4">
          <Link href="/training">
            <a className="text-sm text-gray-700 hover:text-primary-600">Training</a>
          </Link>
          <Link href="/live-prediction">
            <a className="text-sm text-gray-700 hover:text-primary-600">Live Prediction</a>
          </Link>
          <Link href="/upload">
            <a className="text-sm text-gray-700 hover:text-primary-600">Upload</a>
          </Link>
        </div>
      </div>
    </div>
  )
}
