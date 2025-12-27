import Link from 'next/link'
import { useRouter } from 'next/router'

const Item = ({ href, label, icon }) => {
  const router = useRouter()
  const active = router.pathname === href
  return (
    <Link href={href}>
      <a className={`flex items-center px-3 py-2 rounded-md text-sm ${active ? 'bg-primary-50 text-primary-700' : 'text-gray-700 hover:bg-gray-100'}`}>
        <span className="mr-2">{icon}</span>
        {label}
      </a>
    </Link>
  )
}

export default function Sidebar() {
  return (
    <div className="w-60 border-r bg-white h-full">
      <div className="p-3">
        <Item href="/" label="Beranda" icon="🏠" />
        <Item href="/upload" label="Upload Data" icon="⬆️" />
        <Item href="/training" label="Training Model" icon="🧪" />
        <Item href="/sessions" label="Daftar Sesi" icon="🗂️" />
        <Item href="/live-prediction" label="Live Prediction" icon="⚡" />
        <Item href="/results" label="Analisis Hasil" icon="📊" />
        <Item href="/about" label="Tentang" icon="ℹ️" />
      </div>
    </div>
  )
}
