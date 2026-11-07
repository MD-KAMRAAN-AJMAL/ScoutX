import { useEffect, useState } from 'react'
import { IndexingStatusCard } from './components/IndexingStatus'
import { FilesPage } from './pages/FilesPage'
import { SearchPage } from './pages/SearchPage'
import { SettingsPage } from './pages/SettingsPage'
import { api, type IndexStatus } from './services/api'

type Page = 'search' | 'files' | 'settings'

const navigation: Array<{ id: Page; label: string; icon: string }> = [
  { id: 'search', label: 'Search', icon: '⌕' },
  { id: 'files', label: 'Files', icon: '▤' },
  { id: 'settings', label: 'Settings', icon: '⚙' },
]

export default function App() {
  const [page, setPage] = useState<Page>('search')
  const [apiOnline, setApiOnline] = useState(false)
  const [indexStatus, setIndexStatus] = useState<IndexStatus>()
  const [indexStatusUnavailable, setIndexStatusUnavailable] = useState(false)

  useEffect(() => {
    let active = true
    api.health().then(() => { if (active) setApiOnline(true) }).catch(() => { if (active) setApiOnline(false) })
    api.indexStatus().then((status) => {
      if (active) setIndexStatus(status)
    }).catch(() => {
      if (active) setIndexStatusUnavailable(true)
    })
    return () => { active = false }
  }, [])

  return (
    <div className="min-h-screen bg-[#f7f8f6] text-[#17211f]">
      <aside className="fixed inset-y-0 left-0 z-10 hidden w-[248px] flex-col border-r border-[#e9ede9] bg-[#fbfcfa] px-4 py-5 lg:flex">
        <button onClick={() => setPage('search')} className="mb-9 flex items-center gap-3 px-2 text-left">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-[#1f5e47] text-lg font-semibold text-white">s</span>
          <span><span className="block text-[15px] font-semibold tracking-tight">scout<span className="text-[#4b906d]">x</span></span><span className="block text-[10px] tracking-wide text-[#a0aaa3]">YOUR FILES, IN FOCUS</span></span>
        </button>
        <nav className="space-y-1" aria-label="Main navigation">
          {navigation.map((item) => <button key={item.id} onClick={() => setPage(item.id)} className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition ${page === item.id ? 'bg-[#eaf2ed] font-medium text-[#246246]' : 'text-[#77827b] hover:bg-[#f1f4f1] hover:text-[#3d4c43]'}`}><span className="w-5 text-center text-base">{item.icon}</span>{item.label}</button>)}
        </nav>
        <div className="mt-auto"><IndexingStatusCard status={indexStatus} unavailable={!apiOnline || indexStatusUnavailable} /><div className="mt-4 flex items-center gap-2 px-2 text-[11px] text-[#9aa39d]"><span className={`h-1.5 w-1.5 rounded-full ${apiOnline ? 'bg-[#59a27b]' : 'bg-[#d2a259]'}`} />{apiOnline ? 'API connected' : 'Connecting to local API'}</div></div>
      </aside>
      <div className="lg:pl-[248px]">
        <header className="sticky top-0 z-[5] flex h-[66px] items-center justify-between border-b border-[#e9ede9] bg-[#f9faf8]/90 px-5 backdrop-blur sm:px-8">
          <div className="flex items-center gap-2 lg:hidden"><span className="grid h-8 w-8 place-items-center rounded-lg bg-[#1f5e47] font-semibold text-white">s</span><span className="font-semibold">scout<span className="text-[#4b906d]">x</span></span></div>
          <div className="hidden text-sm text-[#9aa39d] lg:block">{page === 'search' ? 'Search your workspace' : page === 'files' ? 'Your file library' : 'Workspace settings'}</div>
          <nav className="flex items-center gap-1 lg:hidden" aria-label="Mobile navigation">{navigation.map((item) => <button key={item.id} aria-label={item.label} onClick={() => setPage(item.id)} className={`rounded-lg px-3 py-2 text-sm ${page === item.id ? 'bg-[#eaf2ed] text-[#246246]' : 'text-[#77827b]'}`}>{item.icon}</button>)}</nav>
          <div className="flex items-center gap-2 text-xs text-[#7c8980]"><span className={`h-2 w-2 rounded-full ${apiOnline ? 'bg-[#55a77b]' : 'bg-[#d6a65b]'}`} />Local workspace</div>
        </header>
        <main>{page === 'search' ? <SearchPage /> : page === 'files' ? <FilesPage /> : <SettingsPage />}</main>
      </div>
    </div>
  )
}
