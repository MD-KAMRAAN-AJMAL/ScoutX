import type { FileRecord } from '../services/api'

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
        <p className="mt-1 text-sm text-[#89948d]">Connect a folder in Settings to get started.</p>
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
          </div>
          <span className="rounded-full bg-[#f1f4f2] px-3 py-1 text-xs text-[#75827a]">{file.indexing_status}</span>
        </article>
      ))}
    </div>
  )
}
