import { useEffect, useState } from 'react'
import { FileList } from '../components/FileList'
import { api, APIRequestError, type FileRecord } from '../services/api'

export function FilesPage() {
  const [files, setFiles] = useState<FileRecord[]>([])
  const [notice, setNotice] = useState('')

  useEffect(() => {
    api.files().then((response) => setFiles(response.items)).catch((error: unknown) => {
      setNotice(error instanceof APIRequestError ? error.message : 'Could not load files.')
    })
  }, [])

  return <div className="mx-auto max-w-5xl px-6 py-12 sm:px-10"><p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#71917e]">Library</p><h1 className="mt-2 text-3xl font-semibold tracking-tight text-[#1d2923]">Your files</h1><p className="mt-2 text-sm text-[#849088]">Browse files discovered in your authorized folders.</p>{notice && <div role="status" className="mt-7 rounded-xl border border-[#e7e9dd] bg-[#fcfbf6] px-4 py-3 text-sm text-[#706a52]">{notice}</div>}<div className="mt-8"><FileList files={files} message={notice ? 'File listing is not available yet' : undefined} /></div></div>
}
