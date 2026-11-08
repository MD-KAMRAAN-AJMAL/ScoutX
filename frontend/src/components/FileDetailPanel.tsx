import { useEffect, useState } from 'react'
import { api, APIRequestError, type FileDetailResponse } from '../services/api'

interface FileDetailPanelProps {
  fileId: string
  highlightChunkId?: string
  query?: string
  onClose: () => void
}

function sourceLocation(snippet: FileDetailResponse['snippets'][number]): string {
  if (snippet.page_number) return `Page ${snippet.page_number}`
  if (snippet.line_start) return `Lines ${snippet.line_start}${snippet.line_end && snippet.line_end !== snippet.line_start ? `–${snippet.line_end}` : ''}`
  if (snippet.start_time_ms != null) {
    const seconds = Math.floor(snippet.start_time_ms / 1000)
    return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`
  }
  return snippet.section_title ?? `Chunk ${snippet.chunk_id.slice(0, 8)}`
}

export function FileDetailPanel({ fileId, highlightChunkId, query, onClose }: FileDetailPanelProps) {
  const [detail, setDetail] = useState<FileDetailResponse>()
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError('')
    api.fileDetail(fileId, controller.signal).then((value) => {
      setDetail(value)
      void api.feedback(fileId, 'opened', query).catch(() => undefined)
    }).catch((reason: unknown) => {
      if (reason instanceof DOMException && reason.name === 'AbortError') return
      setError(reason instanceof APIRequestError ? reason.message : 'File details are unavailable.')
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => controller.abort()
  }, [fileId, query])

  return (
    <aside aria-label="File details" className="rounded-2xl border border-[#dfe7e1] bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#71917e]">File details</p>
          <h2 className="mt-2 break-words text-lg font-semibold text-[#27352e]">{detail?.file.name ?? 'Selected file'}</h2>
          {detail && <p className="mt-1 break-all text-xs text-[#89948d]">{detail.file.path}</p>}
        </div>
        <button type="button" onClick={onClose} aria-label="Close file details" className="rounded-lg px-2 py-1 text-lg text-[#77827b] hover:bg-[#f1f4f1]">×</button>
      </div>
      {loading && <p role="status" className="mt-6 text-sm text-[#89948d]">Loading details…</p>}
      {error && <p role="alert" className="mt-5 rounded-xl bg-[#fff0ed] p-3 text-sm text-[#a44f40]">{error}</p>}
      {detail && <>
        <dl className="mt-5 grid grid-cols-2 gap-3 rounded-xl bg-[#f7f9f7] p-4 text-xs">
          <div><dt className="text-[#89948d]">Type</dt><dd className="mt-1 font-medium text-[#405048]">{detail.file.mime_type ?? detail.file.extension ?? 'Unknown'}</dd></div>
          <div><dt className="text-[#89948d]">Size</dt><dd className="mt-1 font-medium text-[#405048]">{detail.file.size_bytes.toLocaleString()} bytes</dd></div>
          <div><dt className="text-[#89948d]">Indexing</dt><dd className="mt-1 font-medium capitalize text-[#405048]">{detail.file.indexing_status}</dd></div>
          <div><dt className="text-[#89948d]">Version</dt><dd className="mt-1 font-medium text-[#405048]">{detail.file.version}</dd></div>
          <div className="col-span-2"><dt className="text-[#89948d]">Last indexed</dt><dd className="mt-1 font-medium text-[#405048]">{detail.file.last_indexed_at ? new Date(detail.file.last_indexed_at).toLocaleString() : 'Not indexed yet'}</dd></div>
        </dl>
        <div className="mt-6">
          <h3 className="text-sm font-semibold text-[#334139]">Indexed evidence</h3>
          {!detail.snippets.length && <p className="mt-3 rounded-xl border border-dashed border-[#dce4de] p-4 text-sm text-[#89948d]">No active extracted snippets are available.</p>}
          <ul className="mt-3 space-y-3">
            {detail.snippets.map((snippet) => <li key={snippet.chunk_id} id={`chunk-${snippet.chunk_id}`} className={`rounded-xl border p-4 ${snippet.chunk_id === highlightChunkId ? 'border-[#72ae8d] bg-[#f0f7f2]' : 'border-[#edf0ee] bg-[#fbfcfb]'}`}>
              <div className="mb-2 flex items-center justify-between gap-2 text-[11px] text-[#859189]"><span>{sourceLocation(snippet)}</span><span>{snippet.modality}</span></div>
              <p className="whitespace-pre-wrap break-words text-sm leading-6 text-[#46534b]">{snippet.snippet}</p>
            </li>)}
          </ul>
        </div>
      </>}
    </aside>
  )
}
