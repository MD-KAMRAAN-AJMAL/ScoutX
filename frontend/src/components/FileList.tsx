import type { FileRecord } from '../services/api'

const statusStyles: Record<FileRecord['indexing_status'], string> = {
  pending: 'bg-[#f1f4f2] text-[#75827a]',
  processing: 'bg-[#eaf1fb] text-[#496b9d]',
  indexed: 'bg-[#eaf4ee] text-[#367154]',
  partial: 'bg-[#fff4df] text-[#926b24]',
  failed: 'bg-[#fff0ed] text-[#a44f40]',
  unsupported: 'bg-[#f2eff8] text-[#725c92]',
  deleted: 'bg-[#f0f1f1] text-[#7c8380]',
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let size = bytes / 1024
  let unit = 0
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024
    unit += 1
  }
  return `${size.toFixed(size >= 10 ? 0 : 1)} ${units[unit]}`
}

function formatDate(value?: string | null): string {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString()
}

interface FileListProps {
  files?: FileRecord[]
  message?: string
}

export function FileList({ files = [], message }: FileListProps) {
  if (!files.length) {
    return (
      <div className="rounded-2xl border border-dashed border-[#dce4de] bg-white/70 px-6 py-12 text-center">
        <div className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-2xl bg-[#e8f3ed] text-xl text-[#28664e]">▤</div>
        <p className="font-medium text-[#34423b]">{message ?? 'Your indexed files will show up here'}</p>
        <p className="mt-1 text-sm text-[#89948d]">Authorize a folder above to get started.</p>
      </div>
    )
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-[#e3e8e4] bg-white">
      {files.map((file) => (
        <article key={file.id} className="flex items-center gap-4 border-b border-[#edf0ee] px-5 py-4 last:border-0">
          <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-[#eff4f0] text-sm font-semibold uppercase text-[#537262]">{file.name.split('.').pop()?.slice(0, 3) ?? 'FILE'}</span>
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-sm font-semibold text-[#27352e]">{file.name}</h3>
            <p className="truncate text-xs text-[#929c95]">{file.path}</p>
            <p className="mt-1 text-xs text-[#929c95]">{file.mime_type ?? 'Unknown type'} · {formatSize(file.size_bytes)} · Modified {formatDate(file.modified_at)}</p>
            {file.extraction_error && <p className="mt-1 break-words text-xs text-[#a44f40]">Extraction error: {file.extraction_error}</p>}
          </div>
          <span className={`shrink-0 rounded-full px-3 py-1 text-xs capitalize ${statusStyles[file.indexing_status]}`}>{file.indexing_status}</span>
        </article>
      ))}
    </div>
  )
}
