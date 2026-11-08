import { useEffect, useState, type FormEvent } from 'react'
import { IndexingStatusCard } from '../components/IndexingStatus'
import { PersonalizationSettings } from '../components/PersonalizationSettings'
import { api, APIRequestError, type IndexStatus, type RootRecord } from '../services/api'

export function SettingsPage() {
  const [roots, setRoots] = useState<RootRecord[]>([])
  const [indexStatus, setIndexStatus] = useState<IndexStatus>()
  const [notice, setNotice] = useState('')
  const [rootPath, setRootPath] = useState('')
  const [savingRoot, setSavingRoot] = useState(false)

  useEffect(() => {
    Promise.all([api.roots(), api.indexStatus()]).then(([rootList, status]) => {
      setRoots(rootList.items)
      setIndexStatus(status)
    }).catch((error: unknown) => {
      setNotice(error instanceof APIRequestError ? error.message : 'Settings could not be loaded.')
    })
  }, [])

  async function addRoot(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const path = rootPath.trim()
    if (!path || savingRoot) return
    setSavingRoot(true)
    setNotice('')
    try {
      const root = await api.addRoot(path)
      setRoots((existing) => [...existing, root])
      setRootPath('')
    } catch (error) {
      setNotice(error instanceof APIRequestError ? error.message : 'Folder could not be registered.')
    } finally {
      setSavingRoot(false)
    }
  }

  return <div className="mx-auto max-w-5xl px-6 py-12 sm:px-10">
    <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#71917e]">Workspace</p>
    <h1 className="mt-2 text-3xl font-semibold tracking-tight text-[#1d2923]">Settings</h1>
    <p className="mt-2 text-sm text-[#849088]">Manage authorized folders and how ScoutX personalizes recommendations.</p>
    {notice && <div role="status" className="mt-7 rounded-xl border border-[#e7e9dd] bg-[#fcfbf6] px-4 py-3 text-sm text-[#706a52]">{notice}</div>}
    <div className="mt-8 grid gap-5 lg:grid-cols-[1.4fr_1fr]">
      <section className="rounded-2xl border border-[#e3e8e4] bg-white p-5">
        <h2 className="text-sm font-semibold text-[#334139]">Authorized folders</h2>
        <p className="mt-1 text-xs text-[#8b968f]">ScoutX only indexes folders you add here.</p>
        <form onSubmit={addRoot} className="mt-5 flex flex-col gap-2 sm:flex-row">
          <label htmlFor="root-path" className="sr-only">Folder path</label>
          <input id="root-path" value={rootPath} onChange={(event) => setRootPath(event.target.value)} placeholder="/path/to/folder" className="min-w-0 flex-1 rounded-xl border border-[#e2e8e3] bg-[#fbfcfb] px-3 py-2.5 text-sm outline-none focus:border-[#9bcdb7]" />
          <button disabled={!rootPath.trim() || savingRoot} className="rounded-xl bg-[#1c5b45] px-4 py-2.5 text-sm font-semibold text-white hover:bg-[#164b38] disabled:cursor-not-allowed disabled:opacity-45">{savingRoot ? 'Adding…' : 'Add folder'}</button>
        </form>
        {roots.length ? <ul className="mt-5 space-y-3">{roots.map((root) => <li key={root.id} className="rounded-xl bg-[#f7f9f7] p-4"><p className="text-sm font-medium text-[#334139]">{root.display_name ?? root.path}</p><p className="mt-1 break-all text-xs text-[#8b968f]">{root.path}</p></li>)}</ul> : <div className="mt-5 rounded-xl border border-dashed border-[#dce4de] px-4 py-8 text-center text-sm text-[#89948d]">{notice ? 'Folder management is not available yet.' : 'No folders connected.'}</div>}
        <p className="mt-4 text-xs leading-5 text-[#9aa39d]">You authorize each folder before ScoutX indexes it.</p>
      </section>
      <IndexingStatusCard status={indexStatus} unavailable={Boolean(notice)} />
    </div>
    <PersonalizationSettings roots={roots} />
  </div>
}
