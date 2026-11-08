import { useState, type FormEvent } from 'react'

interface SearchBarProps {
  onSearch: (query: string) => void
  busy?: boolean
  onCancel?: () => void
  placeholder?: string
}

export function SearchBar({ onSearch, busy = false, onCancel, placeholder = 'Search files, ideas, or a phrase…' }: SearchBarProps) {
  const [query, setQuery] = useState('')

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const trimmed = query.trim()
    if (trimmed && !busy) onSearch(trimmed)
  }

  return (
    <form onSubmit={submit} className="flex w-full items-center gap-3 rounded-2xl border border-[#e3e8e4] bg-white p-2 shadow-[0_8px_28px_rgba(28,53,43,0.07)] focus-within:border-[#9bcdb7]">
      <span aria-hidden="true" className="pl-3 text-xl text-[#75827b]">⌕</span>
      <input
        aria-label="Search your files"
        className="min-w-0 flex-1 bg-transparent py-3 text-[15px] text-[#17211f] outline-none placeholder:text-[#9aa59f]"
        placeholder={placeholder}
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      <kbd className="hidden rounded-md border border-[#e7ebe8] px-2 py-1 text-xs text-[#8a958e] sm:inline">↵</kbd>
      {busy && <span role="status" aria-live="polite" className="px-2 text-sm font-medium text-[#59675f]">Searching…</span>}
      {busy && onCancel
        ? <button type="button" onClick={onCancel} className="rounded-xl border border-[#e3e8e4] px-4 py-3 text-sm font-medium text-[#59675f] hover:bg-[#f5f8f5]">Cancel</button>
        : <button disabled={!query.trim() || busy} className="rounded-xl bg-[#1c5b45] px-5 py-3 text-sm font-semibold text-white transition hover:bg-[#164b38] disabled:cursor-not-allowed disabled:opacity-45">{busy ? 'Searching…' : 'Search'}</button>}
    </form>
  )
}
