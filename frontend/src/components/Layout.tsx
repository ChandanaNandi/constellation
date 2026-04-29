import { Outlet, Link } from 'react-router-dom'
import { Sparkles, Database, Brain, Gauge, Settings } from 'lucide-react'

const navigation = [
  { name: 'Dashboard', href: '/', icon: Sparkles },
  { name: 'Data Engine', href: '/data', icon: Database },
  { name: 'Models', href: '/models', icon: Brain },
  { name: 'Shadow Mode', href: '/shadow', icon: Gauge },
]

export default function Layout() {
  return (
    <div className="min-h-screen bg-slate-50">
      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 w-64 bg-constellation-950 text-white">
        <div className="flex items-center gap-2 px-6 py-5 border-b border-constellation-800">
          <Sparkles className="h-8 w-8 text-constellation-400" />
          <span className="text-xl font-bold">Constellation</span>
        </div>

        <nav className="mt-6 px-3">
          {navigation.map((item) => (
            <Link
              key={item.name}
              to={item.href}
              className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium text-constellation-200 hover:bg-constellation-800 hover:text-white transition-colors"
            >
              <item.icon className="h-5 w-5" />
              {item.name}
            </Link>
          ))}
        </nav>

        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-constellation-800">
          <Link
            to="/settings"
            className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium text-constellation-300 hover:bg-constellation-800 hover:text-white transition-colors"
          >
            <Settings className="h-5 w-5" />
            Settings
          </Link>
        </div>
      </div>

      {/* Main content */}
      <div className="pl-64">
        <main className="p-8">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
