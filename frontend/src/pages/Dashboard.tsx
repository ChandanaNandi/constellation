import { useState, useEffect } from 'react'
import { Activity, Image, Cpu, Clock } from 'lucide-react'
import { api } from '../api/client'

interface HealthStatus {
  status: string
  app: string
  version: string
  timestamp: string
}

const stats = [
  { name: 'Images Processed', value: '0', icon: Image, color: 'bg-blue-500' },
  { name: 'Model Accuracy', value: '--', icon: Cpu, color: 'bg-green-500' },
  { name: 'Avg Latency', value: '--', icon: Clock, color: 'bg-yellow-500' },
  { name: 'Disagreements', value: '0', icon: Activity, color: 'bg-red-500' },
]

export default function Dashboard() {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.getHealth()
      .then(setHealth)
      .catch((err) => setError(err.message))
  }, [])

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900">Dashboard</h1>
        <p className="mt-2 text-slate-600">
          Multi-task vision system for autonomous driving scenes
        </p>
      </div>

      {/* Backend Status */}
      <div className="mb-8 p-4 rounded-lg bg-white shadow-sm border border-slate-200">
        <h2 className="text-sm font-medium text-slate-500 mb-2">Backend Status</h2>
        {error ? (
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-red-500" />
            <span className="text-red-600">Disconnected: {error}</span>
          </div>
        ) : health ? (
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-green-500 animate-pulse" />
            <span className="text-green-600">
              {health.app} v{health.version} — {health.status}
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-yellow-500 animate-pulse" />
            <span className="text-yellow-600">Connecting...</span>
          </div>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat) => (
          <div
            key={stat.name}
            className="bg-white rounded-xl shadow-sm border border-slate-200 p-6"
          >
            <div className="flex items-center gap-4">
              <div className={`${stat.color} p-3 rounded-lg`}>
                <stat.icon className="h-6 w-6 text-white" />
              </div>
              <div>
                <p className="text-sm font-medium text-slate-500">{stat.name}</p>
                <p className="text-2xl font-bold text-slate-900">{stat.value}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="flex items-center justify-center gap-2 px-4 py-3 bg-constellation-600 text-white rounded-lg hover:bg-constellation-700 transition-colors">
            <Image className="h-5 w-5" />
            Upload Images
          </button>
          <button className="flex items-center justify-center gap-2 px-4 py-3 bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200 transition-colors">
            <Cpu className="h-5 w-5" />
            Run Inference
          </button>
          <button className="flex items-center justify-center gap-2 px-4 py-3 bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200 transition-colors">
            <Activity className="h-5 w-5" />
            View Shadow Mode
          </button>
        </div>
      </div>
    </div>
  )
}
