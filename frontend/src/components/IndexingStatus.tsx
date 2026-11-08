import type { IndexStatus } from '../services/api'

const statusLabels = [
  ['pending', 'Pending', 'text-[#75827a]'],
  ['processing', 'Processing', 'text-[#496b9d]'],
  ['indexed', 'Fully indexed', 'text-[#367154]'],
  ['partial', 'Partially indexed', 'text-[#926b24]'],
  ['failed', 'Failed', 'text-[#a44f40]'],
  ['unsupported', 'Unsupported', 'text-[#725c92]'],
  ['deleted', 'Deleted', 'text-[#7c8380]'],
] as const

interface IndexingStatusProps {
  status?: IndexStatus
  unavailable?: boolean
}

export function IndexingStatusCard({ status, unavailable = false }: IndexingStatusProps) {
  const indexed = status?.files_indexed ?? 0
  const discovered = status?.files_discovered ?? 0
  const progress = discovered ? Math.min(100, Math.round((indexed / discovered) * 100)) : 0
  const indicator = unavailable || status?.state === 'failed'
    ? 'bg-[#d87868]'
    : status?.state === 'pending' || status?.state === 'processing' || status?.state === 'partial'
      ? 'bg-[#d7a65a]'
      : 'bg-[#54a77e]'

  return (
    <section className="rounded-2xl border border-[#e3e8e4] bg-white p-5">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-semibold text-[#334139]">Indexing status</p>
          <p className="mt-1 text-xs text-[#8b968f]">{unavailable ? 'Status unavailable' : status?.state ?? 'Loading indexing status'}</p>
        </div>
        <span className={`mt-1 h-2.5 w-2.5 rounded-full ${indicator}`} />
      </div>
      <div className="mt-5 h-1.5 overflow-hidden rounded-full bg-[#edf1ee]">
        <div className="h-full rounded-full bg-[#4e9a75] transition-all" style={{ width: `${progress}%` }} />
      </div>
      <div className="mt-3 flex justify-between text-xs text-[#859189]">
        <span>{indexed.toLocaleString()} indexed</span>
        <span>{discovered.toLocaleString()} discovered</span>
      </div>
      {status && <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 border-t border-[#edf0ee] pt-4 sm:grid-cols-4">
        {statusLabels.map(([key, label, color]) => <div key={key} className="flex justify-between gap-2 text-xs"><span className={color}>{label}</span><span className="font-medium text-[#526057]">{status.status_counts[key].toLocaleString()}</span></div>)}
      </div>}
      {!!status?.recent_jobs?.length && <div className="mt-4 border-t border-[#edf0ee] pt-4">
        <p className="mb-2 text-xs font-semibold text-[#68766d]">Recent indexing jobs</p>
        <ul className="space-y-2">
          {status.recent_jobs?.slice(0, 3).map((job) => <li key={job.id} className="rounded-lg bg-[#f7f9f7] px-3 py-2">
            <div className="flex items-center justify-between gap-2 text-xs">
              <span className="truncate text-[#536158]">{job.stage ?? job.job_type.replaceAll('_', ' ')}</span>
              <span className={job.status === 'failed' ? 'shrink-0 text-[#a44f40]' : job.status === 'running' ? 'shrink-0 text-[#496b9d]' : 'shrink-0 text-[#75827a]'}>{job.status}{job.progress_percent != null ? ` · ${job.progress_percent}%` : ''}</span>
            </div>
            {job.error_message && <p className="mt-1 line-clamp-2 text-xs text-[#a44f40]">{job.error_message}</p>}
          </li>)}
        </ul>
      </div>}
    </section>
  )
}
