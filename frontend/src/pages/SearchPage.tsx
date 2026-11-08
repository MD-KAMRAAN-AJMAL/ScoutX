import { useEffect, useRef, useState } from 'react'
import { FileDetailPanel } from '../components/FileDetailPanel'
import { SearchBar } from '../components/SearchBar'
import { SearchResults } from '../components/SearchResults'
import { RecommendationsPanel } from '../components/RecommendationsPanel'
import { api, APIRequestError, type AnswerCitation, type FileRecommendation, type RootRecord, type SearchMode, type SearchResponse } from '../services/api'

function answerWithCitations(answer: string, citations: AnswerCitation[], open: (citation: AnswerCitation) => void) {
  const byId = new Map(citations.flatMap((citation) => citation.citation_ids.map((id) => [id, citation] as const)))
  return answer.split(/(\[S\d+\])/g).map((part, index) => {
    const match = /^\[(S\d+)\]$/.exec(part)
    const citation = match ? byId.get(match[1]) : undefined
    return citation
      ? <button key={`${part}-${index}`} type="button" onClick={() => open(citation)} className="mx-0.5 rounded bg-[#eaf4ee] px-1.5 py-0.5 align-baseline text-xs font-semibold text-[#28664e] hover:bg-[#d9ecdf]" aria-label={`Open citation ${part}`}>{part}</button>
      : <span key={`text-${index}`}>{part}</span>
  })
}

export function SearchPage() {
  const [busy, setBusy] = useState(false)
  const [mode, setMode] = useState<SearchMode>('auto')
  const [generateAnswer, setGenerateAnswer] = useState(true)
  const [extensionFilter, setExtensionFilter] = useState('')
  const [rootFilter, setRootFilter] = useState('')
  const [modifiedAfter, setModifiedAfter] = useState('')
  const [modifiedBefore, setModifiedBefore] = useState('')
  const [roots, setRoots] = useState<RootRecord[]>([])
  const [response, setResponse] = useState<SearchResponse>()
  const [notice, setNotice] = useState('')
  const [selectedFile, setSelectedFile] = useState<FileRecommendation>()
  const [highlightChunkId, setHighlightChunkId] = useState<string>()
  const controller = useRef<AbortController | undefined>(undefined)

  useEffect(() => {
    const request = new AbortController()
    api.roots(request.signal).then((value) => setRoots(value.items)).catch(() => undefined)
    return () => request.abort()
  }, [])

  async function search(query: string) {
    controller.current?.abort()
    const current = new AbortController()
    controller.current = current
    setBusy(true)
    setResponse(undefined)
    setSelectedFile(undefined)
    setNotice('')
    try {
      const extensions = extensionFilter.split(',').map((item) => item.trim()).filter(Boolean)
      const result = await api.search({
        query, limit: 20, mode, generate_answer: generateAnswer,
        extensions,
        root_id: rootFilter || undefined,
        modified_after: modifiedAfter ? `${modifiedAfter}T00:00:00Z` : undefined,
        modified_before: modifiedBefore ? `${modifiedBefore}T23:59:59Z` : undefined,
      }, current.signal)
      if (controller.current === current) setResponse(result)
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return
      if (controller.current !== current) return
      setNotice(error instanceof APIRequestError ? error.message : 'Search could not be completed.')
    } finally {
      if (controller.current === current) setBusy(false)
    }
  }

  function cancel() {
    controller.current?.abort()
    controller.current = undefined
    setBusy(false)
  }

  function openCitation(citation: AnswerCitation) {
    setSelectedFile({
      file_id: citation.file_id,
      file_name: citation.file_name,
      file_path: citation.file_path,
      score: citation.relevance_score,
      scoring_method: 'citation',
      retrieval_sources: [],
      supporting_chunks: [],
    })
    setHighlightChunkId(citation.locations[0]?.chunk_id ?? citation.chunk_id ?? undefined)
  }

  function selectFile(file: FileRecommendation, chunkId?: string) {
    setSelectedFile(file)
    setHighlightChunkId(chunkId)
  }

  const generationNotice = response?.generation_status === 'insufficient_evidence'
    ? 'The retrieved evidence is weak. Review the linked files before relying on this answer.'
    : response?.generation_status === 'llm_error'
      ? 'Answer generation was unavailable. The retrieved files are still available below.'
      : response?.reranking_status === 'fallback' || response?.reranking_status === 'disabled_fallback'
        ? 'The reranker was unavailable or disabled; results use the retrieval ranking.'
        : ''

  return (
    <div className="mx-auto max-w-6xl px-6 pb-16 pt-12 sm:px-10 sm:pt-16">
      <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-[#dce9e1] bg-[#edf5f0] px-3 py-1.5 text-xs font-medium text-[#367154]"><span className="h-1.5 w-1.5 rounded-full bg-[#56a27d]" />Your files, understood</div>
      <h1 className="max-w-2xl text-4xl font-semibold leading-tight tracking-[-0.04em] text-[#17211f] sm:text-5xl">Find the file you’re <span className="text-[#3d8061]">looking for.</span></h1>
      <p className="mt-4 max-w-xl text-base leading-7 text-[#7e8982]">Search by name, topic, or a detail you remember. Your files stay on your device.</p>
      <div className="mt-8 flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-2 text-xs text-[#738078]">Search mode
          <select value={mode} onChange={(event) => setMode(event.target.value as SearchMode)} className="rounded-lg border border-[#e2e8e3] bg-white px-2.5 py-2 text-xs text-[#334139]">
            <option value="auto">Automatic</option><option value="keyword">Keyword</option><option value="semantic">Semantic</option><option value="hybrid">Hybrid</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs text-[#738078]"><input type="checkbox" checked={generateAnswer} onChange={(event) => setGenerateAnswer(event.target.checked)} className="accent-[#1c5b45]" />Generate an answer with citations</label>
      </div>
      <details className="mt-4 max-w-3xl rounded-xl border border-[#e3e8e4] bg-white px-4 py-3">
        <summary className="cursor-pointer text-xs font-medium text-[#68766d]">Filters</summary>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <label className="text-xs text-[#75827a]">Extensions<input value={extensionFilter} onChange={(event) => setExtensionFilter(event.target.value)} placeholder="pdf, .py" className="mt-1 block w-full rounded-lg border border-[#e2e8e3] px-3 py-2 text-sm text-[#334139]" /></label>
          <label className="text-xs text-[#75827a]">Folder<select value={rootFilter} onChange={(event) => setRootFilter(event.target.value)} className="mt-1 block w-full rounded-lg border border-[#e2e8e3] bg-white px-3 py-2 text-sm text-[#334139]"><option value="">All folders</option>{roots.map((root) => <option key={root.id} value={root.id}>{root.display_name ?? root.path}</option>)}</select></label>
          <label className="text-xs text-[#75827a]">Modified after<input type="date" value={modifiedAfter} onChange={(event) => setModifiedAfter(event.target.value)} className="mt-1 block w-full rounded-lg border border-[#e2e8e3] px-3 py-2 text-sm text-[#334139]" /></label>
          <label className="text-xs text-[#75827a]">Modified before<input type="date" value={modifiedBefore} onChange={(event) => setModifiedBefore(event.target.value)} className="mt-1 block w-full rounded-lg border border-[#e2e8e3] px-3 py-2 text-sm text-[#334139]" /></label>
        </div>
      </details>
      <div className="mt-4"><SearchBar onSearch={(query) => { void search(query) }} busy={busy} onCancel={cancel} /></div>
      <div className="mt-3 flex flex-wrap gap-2 text-xs text-[#78857d]"><span className="mr-1 py-2">Try:</span>{['meeting notes from Tuesday', 'the file with budget forecasts', 'main.py'].map((example) => <span key={example} className="rounded-full border border-[#e4e9e5] bg-white px-3 py-2">“{example}”</span>)}</div>
      {!response && <RecommendationsPanel />}

      {notice && <div role="alert" className="mt-8 rounded-xl border border-[#f0d7d2] bg-[#fff7f5] px-4 py-3 text-sm text-[#965748]">{notice}</div>}
      {generationNotice && <div role="status" className="mt-8 rounded-xl border border-[#e7e9dd] bg-[#fcfbf6] px-4 py-3 text-sm text-[#706a52]">{generationNotice}</div>}

      {response && <div className={`mt-8 grid items-start gap-5 ${selectedFile ? 'xl:grid-cols-[minmax(0,1fr)_380px]' : ''}`}>
        <div className="min-w-0 space-y-7">
          <div className="flex flex-wrap items-center justify-between gap-2"><div><p className="text-xs text-[#89948d]">Results for</p><h2 className="mt-1 text-lg font-semibold text-[#27352e]">{response.query}</h2></div><span className="rounded-full bg-[#eef3ef] px-3 py-1.5 text-xs text-[#65776b]">{response.response_mode === 'answer' ? 'Answer + files' : 'Files only'} · {response.files.length} files</span></div>
          {response.response_mode === 'answer' && response.answer && <section aria-label="Grounded answer" className="rounded-2xl border border-[#dce9e1] bg-white p-5 sm:p-6">
            <div className="flex items-center justify-between gap-3"><h2 className="text-sm font-semibold text-[#334139]">Answer from your files</h2><span className="text-[11px] text-[#87948b]">{response.generation_status.replaceAll('_', ' ')}</span></div>
            <p className="mt-4 whitespace-pre-wrap text-sm leading-7 text-[#46534b]">{answerWithCitations(response.answer, response.citations, openCitation)}</p>
            {response.generation_status === 'insufficient_evidence' && <p className="mt-3 text-xs text-[#8a7140]">Evidence may not fully answer your question.</p>}
            {!!response.citations.length && <div className="mt-5 border-t border-[#edf0ee] pt-4"><h3 className="text-xs font-semibold text-[#68766d]">Sources</h3><ul className="mt-2 flex flex-wrap gap-2">{response.citations.map((citation) => <li key={`${citation.file_id}-${citation.citation_id}`}><button type="button" onClick={() => openCitation(citation)} className="rounded-lg border border-[#e2e9e3] bg-[#f8faf8] px-3 py-2 text-left text-xs text-[#496153] hover:bg-[#eff6f1]">{citation.citation_ids.map((id) => `[${id}]`).join(' ')} {citation.file_name}{citation.locations[0]?.page_number ? ` · p. ${citation.locations[0].page_number}` : ''}</button></li>)}</ul></div>}
          </section>}

          <section><div className="mb-3 flex items-center justify-between"><h2 className="text-sm font-semibold text-[#334139]">Matching files</h2><span className="text-xs text-[#89948d]">Scores are for ordering, not calibrated confidence</span></div><SearchResults files={response.files} query={response.query} runId={response.run_id} onSelect={selectFile} /></section>
          {!!response.warnings.length && <details className="rounded-xl border border-[#e7e9dd] bg-[#fcfbf6] px-4 py-3 text-xs text-[#706a52]"><summary className="cursor-pointer font-medium">Some search steps reported issues</summary><ul className="mt-2 list-disc space-y-1 pl-5">{response.warnings.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}</ul></details>}
        </div>
        {selectedFile && <FileDetailPanel fileId={selectedFile.file_id} highlightChunkId={highlightChunkId} query={response.query} onClose={() => setSelectedFile(undefined)} />}
      </div>}

      {!response && !notice && <div className="mt-16 grid gap-4 border-t border-[#e7ebe7] pt-7 sm:grid-cols-3">{[['⌕', 'Search naturally', 'Describe what you remember.'], ['◈', 'Private by design', 'Your data stays local.'], ['↗', 'Grounded results', 'Inspect source snippets and citations.']].map(([icon, title, copy]) => <div key={title} className="flex gap-3"><span className="text-lg text-[#4d8b6a]">{icon}</span><div><p className="text-sm font-medium text-[#405048]">{title}</p><p className="mt-1 text-xs text-[#929c95]">{copy}</p></div></div>)}</div>}
    </div>
  )
}
