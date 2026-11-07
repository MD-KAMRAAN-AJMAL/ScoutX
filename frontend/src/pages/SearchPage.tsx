import { useState } from 'react'
import { FileList } from '../components/FileList'
import { SearchBar } from '../components/SearchBar'
import { api, APIRequestError, type FileRecord } from '../services/api'

export function SearchPage() {
  const [busy, setBusy] = useState(false)
  const [files, setFiles] = useState<FileRecord[]>([])
  const [notice, setNotice] = useState('')

  async function search(query: string) {
    setBusy(true)
    setFiles([])
    setNotice('')
    try {
      const result = await api.search({ query })
      setFiles(result.results.map((item) => ({
        id: item.file_id,
        name: item.name,
        path: item.path,
        indexing_status: 'indexed',
      })))
      setNotice(result.answer ?? `${result.results.length} matching files`)
    } catch (error) {
      setNotice(error instanceof APIRequestError ? error.message : 'Search could not be completed.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-6 pb-16 pt-16 sm:px-10 sm:pt-24">
      <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-[#dce9e1] bg-[#edf5f0] px-3 py-1.5 text-xs font-medium text-[#367154]">
        <span className="h-1.5 w-1.5 rounded-full bg-[#56a27d]" /> Your files, understood
      </div>
      <h1 className="max-w-2xl text-4xl font-semibold leading-tight tracking-[-0.04em] text-[#17211f] sm:text-5xl">Find the file you’re <span className="text-[#3d8061]">looking for.</span></h1>
      <p className="mt-4 max-w-xl text-base leading-7 text-[#7e8982]">Search by name, topic, or a detail you remember. Your files stay on your device.</p>
      <div className="mt-9"><SearchBar onSearch={search} busy={busy} /></div>
      <div className="mt-4 flex flex-wrap gap-2 text-xs text-[#78857d]">
        <span className="mr-1 py-2">Try:</span>
        {['“meeting notes from Tuesday”', '“the file with budget forecasts”', '“main.py”'].map((example) => <span key={example} className="rounded-full border border-[#e4e9e5] bg-white px-3 py-2">{example}</span>)}
      </div>
      {notice && <div role="status" className="mt-10 rounded-xl border border-[#e7e9dd] bg-[#fcfbf6] px-4 py-3 text-sm text-[#706a52]">{notice}</div>}
      {files.length > 0 && <div className="mt-8"><h2 className="mb-3 text-sm font-semibold text-[#334139]">Results</h2><FileList files={files} /></div>}
      <div className="mt-16 grid gap-4 border-t border-[#e7ebe7] pt-7 sm:grid-cols-3">
        {[['⌕', 'Search naturally', 'Describe what you remember.'], ['◈', 'Private by design', 'Your data stays local.'], ['↗', 'Relevant results', 'Clear sources for every match.']].map(([icon, title, copy]) => <div key={title} className="flex gap-3"><span className="text-lg text-[#4d8b6a]">{icon}</span><div><p className="text-sm font-medium text-[#405048]">{title}</p><p className="mt-1 text-xs text-[#929c95]">{copy}</p></div></div>)}
      </div>
    </div>
  )
}
