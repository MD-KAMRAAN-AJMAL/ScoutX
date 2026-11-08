import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { FileList } from '../components/FileList'
import { FileDetailPanel } from '../components/FileDetailPanel'
import { IndexingStatusCard } from '../components/IndexingStatus'
import { api, APIRequestError, type FileRecord, type FileStatus, type IndexStatus, type RootRecord } from '../services/api'

const PAGE_SIZE = 50

export function FilesPage() {
  const [roots, setRoots] = useState<RootRecord[]>([])
  const [files, setFiles] = useState<FileRecord[]>([])
  const [indexStatus, setIndexStatus] = useState<IndexStatus>()
  const [selectedRoot, setSelectedRoot] = useState('all')
  const [path, setPath] = useState('')
  const [offset, setOffset] = useState(0)
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [savingRoot, setSavingRoot] = useState(false)
  const [reconciling, setReconciling] = useState(false)
  const [notice, setNotice] = useState('')
  const [filenameFilter, setFilenameFilter] = useState('')
  const [appliedFilename, setAppliedFilename] = useState('')
  const [extensionFilter, setExtensionFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState<FileStatus | ''>('')
  const [sortBy, setSortBy] = useState<'name' | 'modified_at' | 'size_bytes' | 'last_indexed_at'>('name')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc')
  const [appliedExtension, setAppliedExtension] = useState('')
  const [appliedStatus, setAppliedStatus] = useState<FileStatus | ''>('')
  const [appliedSortBy, setAppliedSortBy] = useState<typeof sortBy>('name')
  const [appliedSortOrder, setAppliedSortOrder] = useState<typeof sortOrder>('asc')
  const [modifiedAfter, setModifiedAfter] = useState('')
  const [modifiedBefore, setModifiedBefore] = useState('')
  const [appliedModifiedAfter, setAppliedModifiedAfter] = useState('')
  const [appliedModifiedBefore, setAppliedModifiedBefore] = useState('')
  const [selectedFileId, setSelectedFileId] = useState<string>()

  const loadFiles = useCallback(async (rootId: string, pageOffset: number) => {
    const result = await api.files({
      limit: PAGE_SIZE,
      offset: pageOffset,
      rootId: rootId === 'all' ? undefined : rootId,
      filenameContains: appliedFilename || undefined,
      extension: appliedExtension || undefined,
      indexingStatus: appliedStatus || undefined,
      sortBy: appliedSortBy,
      sortOrder: appliedSortOrder,
      modifiedAfter: appliedModifiedAfter ? `${appliedModifiedAfter}T00:00:00Z` : undefined,
      modifiedBefore: appliedModifiedBefore ? `${appliedModifiedBefore}T23:59:59Z` : undefined,
    })
    setFiles(result.items)
    setTotal(result.total)
  }, [])

  const loadWorkspace = useCallback(async () => {
    setLoading(true)
    const [rootResult, fileResult, statusResult] = await Promise.allSettled([
      api.roots(),
      api.files({
        limit: PAGE_SIZE,
        offset,
        rootId: selectedRoot === 'all' ? undefined : selectedRoot,
        filenameContains: appliedFilename || undefined,
        extension: appliedExtension || undefined,
        indexingStatus: appliedStatus || undefined,
        sortBy: appliedSortBy,
        sortOrder: appliedSortOrder,
        modifiedAfter: appliedModifiedAfter ? `${appliedModifiedAfter}T00:00:00Z` : undefined,
        modifiedBefore: appliedModifiedBefore ? `${appliedModifiedBefore}T23:59:59Z` : undefined,
      }),
      api.indexStatus(),
    ])
    if (rootResult.status === 'fulfilled') setRoots(rootResult.value.items)
    if (fileResult.status === 'fulfilled') {
      setFiles(fileResult.value.items)
      setTotal(fileResult.value.total)
    }
    if (statusResult.status === 'fulfilled') setIndexStatus(statusResult.value)

    const failed = [rootResult, fileResult, statusResult].find((result) => result.status === 'rejected')
    if (failed?.status === 'rejected') {
      const error: unknown = failed.reason
      setNotice(error instanceof APIRequestError ? error.message : 'Some filesystem data could not be loaded.')
    } else {
      setNotice('')
    }
    setLoading(false)
  }, [appliedExtension, appliedFilename, appliedModifiedAfter, appliedModifiedBefore, appliedSortBy, appliedSortOrder, appliedStatus, offset, selectedRoot])

  useEffect(() => { void loadWorkspace() }, [loadWorkspace])

  async function registerRoot(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const requestedPath = path.trim()
    if (!requestedPath || savingRoot) return
    setSavingRoot(true)
    setNotice('')
    try {
      const root = await api.addRoot(requestedPath)
      setRoots((current) => [...current, root])
      setPath('')
      setSelectedRoot(root.id)
      setOffset(0)
      await loadFiles(root.id, 0)
      const status = await api.indexStatus()
      setIndexStatus(status)
    } catch (error) {
      setNotice(error instanceof APIRequestError ? error.message : 'The directory could not be registered.')
    } finally {
      setSavingRoot(false)
    }
  }

  async function unregisterRoot(root: RootRecord) {
    setNotice('')
    try {
      await api.removeRoot(root.id)
      setRoots((current) => current.map((item) => item.id === root.id ? { ...item, status: 'disabled' } : item))
      if (selectedRoot === root.id) setSelectedRoot('all')
      setNotice('Folder access was removed. Previously discovered file records are retained.')
      void api.indexStatus().then(setIndexStatus).catch(() => undefined)
    } catch (error) {
      setNotice(error instanceof APIRequestError ? error.message : 'Folder access could not be removed.')
    }
  }

  async function reconcile() {
    setReconciling(true)
    setNotice('')
    try {
      const response = await api.reconcile(selectedRoot === 'all' ? undefined : selectedRoot)
      setNotice(`Reconciliation queued for ${response.root_ids.length} ${response.root_ids.length === 1 ? 'folder' : 'folders'}.`)
      const status = await api.indexStatus()
      setIndexStatus(status)
    } catch (error) {
      setNotice(error instanceof APIRequestError ? error.message : 'Reconciliation could not be queued.')
    } finally {
      setReconciling(false)
    }
  }

  function changeRoot(rootId: string) {
    setSelectedRoot(rootId)
    setOffset(0)
  }

  const canReconcile = selectedRoot === 'all'
    ? roots.some((root) => root.status === 'enabled')
    : roots.some((root) => root.id === selectedRoot && root.status === 'enabled')

  return (
    <div className="mx-auto max-w-6xl px-6 py-10 sm:px-10">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#71917e]">Library</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-[#1d2923]">Your files</h1>
          <p className="mt-2 text-sm text-[#849088]">Authorize folders, review discovered files, and track indexing.</p>
        </div>
        <div className="flex gap-2">
          <button onClick={() => void loadWorkspace()} disabled={loading} className="rounded-xl border border-[#dfe7e1] bg-white px-4 py-2.5 text-sm font-semibold text-[#496153] hover:bg-[#f5f8f5] disabled:cursor-wait disabled:opacity-50">Refresh</button>
          <button onClick={reconcile} disabled={reconciling || !canReconcile} className="rounded-xl bg-[#1c5b45] px-4 py-2.5 text-sm font-semibold text-white hover:bg-[#164b38] disabled:cursor-not-allowed disabled:opacity-45">
            {reconciling ? 'Queueing…' : 'Reconcile now'}
          </button>
        </div>
      </div>

      {notice && <div role="status" className="mt-6 rounded-xl border border-[#e7e9dd] bg-[#fcfbf6] px-4 py-3 text-sm text-[#706a52]">{notice}</div>}

      <div className="mt-7 grid gap-5 lg:grid-cols-[1.35fr_1fr]">
        <section className="rounded-2xl border border-[#e3e8e4] bg-white p-5">
          <h2 className="text-sm font-semibold text-[#334139]">Authorized folders</h2>
          <p className="mt-1 text-xs text-[#8b968f]">Only folders you register here can be indexed.</p>
          <form onSubmit={registerRoot} className="mt-5 flex flex-col gap-2 sm:flex-row">
            <label htmlFor="authorized-directory" className="sr-only">Directory path</label>
            <input
              id="authorized-directory"
              value={path}
              onChange={(event) => setPath(event.target.value)}
              placeholder="Enter an absolute directory path"
              autoComplete="off"
              className="min-w-0 flex-1 rounded-xl border border-[#e2e8e3] bg-[#fbfcfb] px-3 py-2.5 text-sm outline-none focus:border-[#9bcdb7]"
            />
            <button disabled={!path.trim() || savingRoot} className="rounded-xl bg-[#1c5b45] px-4 py-2.5 text-sm font-semibold text-white hover:bg-[#164b38] disabled:cursor-not-allowed disabled:opacity-45">
              {savingRoot ? 'Checking…' : 'Authorize folder'}
            </button>
          </form>
          <p className="mt-2 text-xs text-[#9aa39d]">The API checks that this path exists, is a directory, and can be read before saving it.</p>

          <ul className="mt-5 divide-y divide-[#edf0ee]">
            {roots.map((root) => <li key={root.id} className="flex items-center gap-3 py-3 first:pt-0 last:pb-0">
              <span className={`h-2 w-2 shrink-0 rounded-full ${root.status === 'enabled' ? 'bg-[#55a77b]' : 'bg-[#a9b0ac]'}`} />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-[#334139]">{root.display_name ?? root.path}</p>
                <p className="truncate text-xs text-[#8b968f]">{root.path}{root.last_reconciled_at ? ` · Reconciled ${new Date(root.last_reconciled_at).toLocaleString()}` : ''}</p>
              </div>
              <span className="text-xs capitalize text-[#839087]">{root.status}</span>
              {root.status === 'enabled' && <button onClick={() => void unregisterRoot(root)} className="rounded-lg px-2 py-1 text-xs text-[#9b594d] hover:bg-[#fff0ed]">Remove</button>}
            </li>)}
          </ul>
          {!roots.length && !loading && <p className="mt-5 rounded-xl border border-dashed border-[#dce4de] px-4 py-6 text-center text-sm text-[#89948d]">No folders authorized yet.</p>}
        </section>
        <IndexingStatusCard status={indexStatus} unavailable={!indexStatus} />
      </div>

      <section className="mt-9">
        <div className="mb-4 flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
          <div>
            <h2 className="text-lg font-semibold text-[#27352e]">Discovered files</h2>
            <p className="mt-1 text-xs text-[#929c95]">{total.toLocaleString()} file records</p>
          </div>
          <label className="flex items-center gap-2 text-sm text-[#75827a]">Folder
            <select value={selectedRoot} onChange={(event) => changeRoot(event.target.value)} className="rounded-lg border border-[#e2e8e3] bg-white px-3 py-2 text-sm text-[#334139]">
              <option value="all">All folders</option>{roots.map((root) => <option key={root.id} value={root.id}>{root.display_name ?? root.path}</option>)}
            </select>
          </label>
        </div>
        <form onSubmit={(event) => { event.preventDefault(); setOffset(0); setAppliedFilename(filenameFilter.trim()); setAppliedExtension(extensionFilter.trim()); setAppliedStatus(statusFilter); setAppliedSortBy(sortBy); setAppliedSortOrder(sortOrder); setAppliedModifiedAfter(modifiedAfter); setAppliedModifiedBefore(modifiedBefore) }} className="mb-4 grid gap-2 rounded-2xl border border-[#e3e8e4] bg-white p-4 sm:grid-cols-2 xl:grid-cols-5">
          <label className="text-xs text-[#75827a]">Filename<input value={filenameFilter} onChange={(event) => setFilenameFilter(event.target.value)} placeholder="Filter by name" className="mt-1 block w-full rounded-lg border border-[#e2e8e3] px-3 py-2 text-sm text-[#334139]" /></label>
          <label className="text-xs text-[#75827a]">Extension<input value={extensionFilter} onChange={(event) => setExtensionFilter(event.target.value)} placeholder=".pdf, .py" className="mt-1 block w-full rounded-lg border border-[#e2e8e3] px-3 py-2 text-sm text-[#334139]" /></label>
          <label className="text-xs text-[#75827a]">Indexing status<select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as FileStatus | '')} className="mt-1 block w-full rounded-lg border border-[#e2e8e3] bg-white px-3 py-2 text-sm text-[#334139]"><option value="">All statuses</option>{['pending', 'processing', 'indexed', 'partial', 'failed', 'unsupported', 'deleted'].map((status) => <option key={status} value={status}>{status}</option>)}</select></label>
          <label className="text-xs text-[#75827a]">Modified after<input type="date" value={modifiedAfter} onChange={(event) => setModifiedAfter(event.target.value)} className="mt-1 block w-full rounded-lg border border-[#e2e8e3] px-3 py-2 text-sm text-[#334139]" /></label>
          <label className="text-xs text-[#75827a]">Modified before<input type="date" value={modifiedBefore} onChange={(event) => setModifiedBefore(event.target.value)} className="mt-1 block w-full rounded-lg border border-[#e2e8e3] px-3 py-2 text-sm text-[#334139]" /></label>
          <label className="text-xs text-[#75827a]">Sort by<select value={sortBy} onChange={(event) => setSortBy(event.target.value as typeof sortBy)} className="mt-1 block w-full rounded-lg border border-[#e2e8e3] bg-white px-3 py-2 text-sm text-[#334139]"><option value="name">Name</option><option value="modified_at">Modified</option><option value="size_bytes">Size</option><option value="last_indexed_at">Last indexed</option></select></label>
          <div className="flex items-end gap-2"><label className="flex-1 text-xs text-[#75827a]">Order<select value={sortOrder} onChange={(event) => setSortOrder(event.target.value as typeof sortOrder)} className="mt-1 block w-full rounded-lg border border-[#e2e8e3] bg-white px-3 py-2 text-sm text-[#334139]"><option value="asc">Ascending</option><option value="desc">Descending</option></select></label><button className="rounded-lg bg-[#1c5b45] px-4 py-2 text-sm font-semibold text-white">Filter</button></div>
        </form>
        {loading ? <div className="rounded-2xl border border-[#e3e8e4] bg-white px-6 py-12 text-center text-sm text-[#89948d]">Loading files…</div> : <FileList files={files} onSelect={(file) => setSelectedFileId(file.id)} message={notice ? 'File records are unavailable' : 'No files match these filters'} />}
        <div className="mt-4 flex items-center justify-between text-xs text-[#839087]">
          <span>{total ? `${offset + 1}–${Math.min(offset + PAGE_SIZE, total)} of ${total}` : 'No records'}</span>
          <div className="flex gap-2">
            <button disabled={offset === 0 || loading} onClick={() => setOffset((value) => Math.max(0, value - PAGE_SIZE))} className="rounded-lg border border-[#e2e8e3] bg-white px-3 py-1.5 disabled:opacity-40">Previous</button>
            <button disabled={offset + PAGE_SIZE >= total || loading} onClick={() => setOffset((value) => value + PAGE_SIZE)} className="rounded-lg border border-[#e2e8e3] bg-white px-3 py-1.5 disabled:opacity-40">Next</button>
          </div>
        </div>
      </section>
      {selectedFileId && <div className="mt-6 max-w-3xl"><FileDetailPanel fileId={selectedFileId} onClose={() => setSelectedFileId(undefined)} /></div>}
    </div>
  )
}
