import type { IndexStatus } from '../services/api'

interface IndexingStatusProps {
  status?: IndexStatus
  unavailable?: boolean
}

export function IndexingStatusCard({ status, unavailable = false }: IndexingStatusProps) {
  const indexed = status?.files_indexed ?? 0
  const discovered = status?.files_discovered ?? 0
  const progress = discovered ? Math.min(100, Math.round((indexed / discovered) * 100)) : 0

  return (
    <section className="rounded-2xl border border-[#e3e8e4] bg-white p-5">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-semibold text-[#334139]">Indexing status</p>
          <p className="mt-1 text-xs text-[#8b968f]">{unavailable ? 'Status unavailable' : status?.state ?? 'Loading indexing status'}</p>
        </div>
        <span className={`mt-1 h-2.5 w-2.5 rounded-full ${unavailable ? 'bg-[#d7a65a]' : 'bg-[#54a77e]'}`} />
      </div>
      <div className="mt-5 h-1.5 overflow-hidden rounded-full bg-[#edf1ee]">
        <div className="h-full rounded-full bg-[#4e9a75] transition-all" style={{ width: `${progress}%` }} />
      </div>
      <div className="mt-3 flex justify-between text-xs text-[#859189]">
        <span>{indexed.toLocaleString()} indexed</span>
        <span>{discovered.toLocaleString()} discovered</span>
      </div>
    </section>
  )
}
