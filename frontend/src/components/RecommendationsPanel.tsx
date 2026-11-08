import { useEffect, useState } from 'react'
import { api, APIRequestError, type RecommendationItem } from '../services/api'
import { FileDetailPanel } from './FileDetailPanel'

export function RecommendationsPanel() {
  const [items, setItems] = useState<RecommendationItem[]>([])
  const [selectedFileId, setSelectedFileId] = useState<string>()
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const controller = new AbortController()
    api.recommendations('recommended_for_you', 8, undefined, controller.signal)
      .then((result) => setItems(result.items))
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === 'AbortError') return
        setError(reason instanceof APIRequestError ? reason.message : 'Recommendations are unavailable.')
      })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [])

  async function feedback(item: RecommendationItem, action: 'useful' | 'dismissed') {
    await api.feedback(item.file_id, action).catch(() => undefined)
    if (action === 'dismissed') setItems((current) => current.filter((entry) => entry.file_id !== item.file_id))
  }

  function open(item: RecommendationItem) {
    void api.feedback(item.file_id, 'selected').catch(() => undefined)
    setSelectedFileId(item.file_id)
  }

  return <section aria-label="Personalized recommendations" className="mt-10">
    <div className="mb-4 flex items-end justify-between gap-3"><div><p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#71917e]">A good place to pick up</p><h2 className="mt-1 text-lg font-semibold text-[#27352e]">Recommended for you</h2></div><span className="text-xs text-[#89948d]">Uses your local files and preferences when personalization is enabled</span></div>
    {loading && <p role="status" className="rounded-2xl border border-[#e3e8e4] bg-white p-5 text-sm text-[#89948d]">Finding useful files…</p>}
    {error && <p role="status" className="rounded-xl border border-[#e7e9dd] bg-[#fcfbf6] p-4 text-sm text-[#706a52]">{error}</p>}
    {!loading && !error && items.length === 0 && <p className="rounded-2xl border border-dashed border-[#dce4de] bg-white/70 p-5 text-sm text-[#89948d]">No recommendations yet. Add an authorized folder and use ScoutX to build a local activity profile.</p>}
    {!!items.length && <div className="grid gap-3 md:grid-cols-2">{items.map((item) => <article key={item.file_id} className="rounded-2xl border border-[#e3e8e4] bg-white p-4">
      <div className="flex items-start justify-between gap-3"><div className="min-w-0"><button type="button" onClick={() => open(item)} className="truncate text-left text-sm font-semibold text-[#27352e] hover:text-[#246246]">{item.filename}</button><p className="mt-1 truncate text-xs text-[#929c95]">{item.path}</p></div><span className="shrink-0 rounded-full bg-[#eff4f0] px-2 py-1 text-[10px] font-medium text-[#537262]">{item.extension ?? 'file'}</span></div>
      <p className="mt-3 text-xs leading-5 text-[#61736a]">{item.reason}</p>
      <div className="mt-3 flex gap-2"><button type="button" onClick={() => open(item)} className="rounded-lg border border-[#e3e8e4] px-3 py-1.5 text-xs font-medium text-[#496153]">Open details</button><button type="button" onClick={() => void feedback(item, 'useful')} className="rounded-lg px-3 py-1.5 text-xs text-[#61776a] hover:bg-[#f0f7f2]">Useful</button><button type="button" onClick={() => void feedback(item, 'dismissed')} className="rounded-lg px-3 py-1.5 text-xs text-[#89948d] hover:bg-[#f5f6f5]">Dismiss</button></div>
    </article>)}</div>}
    {selectedFileId && <div className="mt-4 max-w-2xl"><FileDetailPanel fileId={selectedFileId} onClose={() => setSelectedFileId(undefined)} /></div>}
  </section>
}
