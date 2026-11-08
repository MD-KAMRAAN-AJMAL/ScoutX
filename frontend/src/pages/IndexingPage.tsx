import { useEffect, useState } from 'react'
import { IndexingStatusCard } from '../components/IndexingStatus'
import { api, APIRequestError, type IndexStatus } from '../services/api'

const REFRESH_MS = 12_000

export function IndexingPage() {
  const [status, setStatus] = useState<IndexStatus>()
  const [error, setError] = useState('')
  const [updatedAt, setUpdatedAt] = useState<Date>()

  useEffect(() => {
    let active = true
    let controller: AbortController | undefined
    let timer: number | undefined

    const refresh = async () => {
      if (!active || document.hidden) return
      controller?.abort()
      controller = new AbortController()
      try {
        const next = await api.indexStatus(controller.signal)
        if (!active) return
        setStatus(next)
        setError('')
        setUpdatedAt(new Date())
      } catch (reason) {
        if (reason instanceof DOMException && reason.name === 'AbortError') return
        if (active) setError(reason instanceof APIRequestError ? reason.message : 'Indexing status is unavailable.')
      } finally {
        if (active) timer = window.setTimeout(() => { void refresh() }, REFRESH_MS)
      }
    }

    const onVisibility = () => {
      if (!document.hidden) {
        if (timer) window.clearTimeout(timer)
        void refresh()
      } else if (timer) {
        window.clearTimeout(timer)
      }
    }

    void refresh()
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      active = false
      controller?.abort()
      if (timer) window.clearTimeout(timer)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [])

  return <div className="mx-auto max-w-5xl px-6 py-10 sm:px-10">
    <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#71917e]">Workspace</p>
    <h1 className="mt-2 text-3xl font-semibold tracking-tight text-[#1d2923]">Indexing activity</h1>
    <p className="mt-2 text-sm text-[#849088]">Status refreshes every 12 seconds while this page is visible.</p>
    {error && <p role="alert" className="mt-5 rounded-xl border border-[#f0d7d2] bg-[#fff7f5] px-4 py-3 text-sm text-[#965748]">{error}</p>}
    {updatedAt && <p className="mt-4 text-xs text-[#929c95]">Updated {updatedAt.toLocaleTimeString()}</p>}
    <div className="mt-5 max-w-2xl"><IndexingStatusCard status={status} unavailable={!status} /></div>
    {status?.last_reconciled_at && <p className="mt-4 text-sm text-[#75827a]">Last directory reconciliation: {new Date(status.last_reconciled_at).toLocaleString()}</p>}
    {!!status?.recent_jobs?.length && <section className="mt-8 rounded-2xl border border-[#e3e8e4] bg-white p-5"><h2 className="text-sm font-semibold text-[#334139]">Recent jobs</h2><ul className="mt-3 divide-y divide-[#edf0ee]">{status.recent_jobs.map((job) => <li key={job.id} className="flex flex-wrap items-center justify-between gap-2 py-3 text-sm"><span className="text-[#536158]">{job.stage ?? job.job_type.replaceAll('_', ' ')}</span><span className="text-xs text-[#7d8a81]">{job.status}{job.progress_percent != null ? ` · ${job.progress_percent}%` : ''}</span>{job.error_message && <p className="w-full text-xs text-[#a44f40]">{job.error_message}</p>}</li>)}</ul></section>}
  </div>
}
