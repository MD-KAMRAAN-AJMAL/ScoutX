import { api, type FileRecommendation, type FeedbackAction } from '../services/api'

interface SearchResultsProps {
  files: FileRecommendation[]
  query: string
  runId?: string
  onSelect: (file: FileRecommendation, chunkId?: string) => void
}

function reference(chunk: FileRecommendation['supporting_chunks'][number]): string {
  if (chunk.page_number) return `Page ${chunk.page_number}`
  if (chunk.line_start) return `Lines ${chunk.line_start}${chunk.line_end && chunk.line_end !== chunk.line_start ? `–${chunk.line_end}` : ''}`
  return chunk.section_title ?? chunk.retrieval_method.replaceAll('_', ' ')
}

export function SearchResults({ files, query, runId, onSelect }: SearchResultsProps) {
  async function feedback(fileId: string, action: FeedbackAction) {
    await api.feedback(fileId, action, query, runId).catch(() => undefined)
  }

  if (!files.length) {
    return <div className="rounded-2xl border border-dashed border-[#dce4de] bg-white/70 px-6 py-12 text-center">
      <p className="font-medium text-[#34423b]">No matching files found</p>
      <p className="mt-1 text-sm text-[#89948d]">Try another phrase, filename, or file detail.</p>
    </div>
  }

  return <div className="space-y-3">
    {files.map((file, index) => {
      const chunk = file.supporting_chunks[0]
      return <article key={file.file_id} className="rounded-2xl border border-[#e3e8e4] bg-white p-5 transition hover:border-[#c7d9cc]">
        <div className="flex items-start gap-4">
          <span aria-hidden="true" className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-[#eff4f0] text-xs font-semibold uppercase text-[#537262]">{file.extension?.replace('.', '').slice(0, 4) ?? 'FILE'}</span>
          <div className="min-w-0 flex-1">
            <button type="button" onClick={() => { void feedback(file.file_id, 'selected'); onSelect(file, chunk?.chunk_id ?? undefined) }} className="text-left text-sm font-semibold text-[#27352e] hover:text-[#246246]">{file.file_name}</button>
            <p className="mt-1 truncate text-xs text-[#929c95]">{file.file_path}</p>
            <p className="mt-1 text-xs text-[#829087]">{file.mime_type ?? file.extension ?? 'File'} · {file.retrieval_sources.join(', ') || 'retrieval'} · Result {index + 1}</p>
            {!!file.personalization_reasons?.length && <p className="mt-2 text-xs text-[#47765c]">{file.personalization_reasons.join(' · ')}</p>}
            {chunk && <div className="mt-3 rounded-xl bg-[#f7f9f7] p-3">
              <p className="mb-1 text-[11px] font-medium text-[#819087]">{reference(chunk)}</p>
              <p className="whitespace-pre-wrap break-words text-sm leading-6 text-[#536158]">{chunk.snippet}</p>
            </div>}
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <button type="button" onClick={() => onSelect(file, chunk?.chunk_id ?? undefined)} className="rounded-lg border border-[#e3e8e4] px-3 py-1.5 text-xs font-medium text-[#496153] hover:bg-[#f5f8f5]">Details</button>
              <button type="button" aria-label={`Mark ${file.file_name} relevant`} onClick={() => void feedback(file.file_id, 'relevant')} className="rounded-lg px-2 py-1.5 text-xs text-[#61776a] hover:bg-[#f0f7f2]">Helpful</button>
              <button type="button" aria-label={`Mark ${file.file_name} not relevant`} onClick={() => void feedback(file.file_id, 'not_relevant')} className="rounded-lg px-2 py-1.5 text-xs text-[#89948d] hover:bg-[#f5f6f5]">Not relevant</button>
            </div>
          </div>
          <span className="shrink-0 rounded-full bg-[#eaf4ee] px-2.5 py-1 text-[11px] font-medium text-[#367154]">Ranked</span>
        </div>
      </article>
    })}
  </div>
}
