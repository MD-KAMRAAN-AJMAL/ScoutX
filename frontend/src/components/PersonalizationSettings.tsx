import { useEffect, useState, type ChangeEvent } from 'react'
import { api, APIRequestError, type ActivityItem, type PersonalizationPreferences, type PersonalizationProfile, type RootRecord } from '../services/api'

interface PersonalizationSettingsProps {
  roots: RootRecord[]
}

function eventLabel(eventType: string): string {
  return ({
    search_performed: 'Searched your files',
    search_result_opened: 'Opened a search result',
    file_opened: 'Opened a file',
    file_previewed: 'Previewed a file',
    opened: 'Opened a file',
    selected: 'Selected a file',
    recommendation_clicked: 'Opened a recommendation',
    recommendation_dismissed: 'Dismissed a recommendation',
    feedback_positive: 'Marked a file useful',
    feedback_negative: 'Marked a file not useful',
  } as Record<string, string>)[eventType] ?? 'Used ScoutX'
}

const defaultPreferences: PersonalizationPreferences = {
  personalization_enabled: true,
  preferred_extensions: [],
  excluded_extensions: [],
  preferred_root_ids: [],
  excluded_root_ids: [],
  recommendation_limit: 12,
}

export function PersonalizationSettings({ roots }: PersonalizationSettingsProps) {
  const [preferences, setPreferences] = useState<PersonalizationPreferences>()
  const [profile, setProfile] = useState<PersonalizationProfile>()
  const [activity, setActivity] = useState<ActivityItem[]>([])
  const [activityOffset, setActivityOffset] = useState(0)
  const [activityTotal, setActivityTotal] = useState(0)
  const [preferredExtensions, setPreferredExtensions] = useState('')
  const [excludedExtensions, setExcludedExtensions] = useState('')
  const [notice, setNotice] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  async function refresh() {
    const results = await Promise.allSettled([
      api.personalizationPreferences(), api.personalizationProfile(), api.activity({ limit: 12, offset: activityOffset }),
    ])
    if (results[0].status === 'fulfilled') {
      setPreferences(results[0].value)
      setPreferredExtensions(results[0].value.preferred_extensions.join(', '))
      setExcludedExtensions(results[0].value.excluded_extensions.join(', '))
    }
    if (results[1].status === 'fulfilled') setProfile(results[1].value)
    if (results[2].status === 'fulfilled') {
      setActivity(results[2].value.items)
      setActivityTotal(results[2].value.total)
    }
    const failed = results.find((result) => result.status === 'rejected')
    if (failed?.status === 'rejected') {
      setNotice(failed.reason instanceof APIRequestError ? failed.reason.message : 'Personalization settings could not be loaded.')
    }
    setLoading(false)
  }

  useEffect(() => { void refresh() }, [])

  async function loadActivity(offset: number) {
    try {
      const result = await api.activity({ limit: 12, offset })
      setActivity(result.items)
      setActivityOffset(offset)
      setActivityTotal(result.total)
    } catch {
      setNotice('Activity history could not be loaded.')
    }
  }

  async function save(next: PersonalizationPreferences) {
    setSaving(true)
    setNotice('')
    try {
      const saved = await api.updatePersonalizationPreferences(next)
      setPreferences(saved)
      setNotice('Personalization settings saved.')
      const refreshedProfile = await api.personalizationProfile().catch(() => undefined)
      if (refreshedProfile) setProfile(refreshedProfile)
    } catch (error) {
      setNotice(error instanceof APIRequestError ? error.message : 'Settings could not be saved.')
    } finally {
      setSaving(false)
    }
  }

  async function clearHistory() {
    if (!window.confirm('Clear your activity and feedback history? This cannot be undone.')) return
    setSaving(true)
    try {
      const result = await api.clearActivity()
      setActivity([])
      setActivityOffset(0)
      setActivityTotal(0)
      const refreshedProfile = await api.personalizationProfile().catch(() => undefined)
      if (refreshedProfile) setProfile(refreshedProfile)
      setNotice(`Cleared ${result.deleted_activity_events} activity events and ${result.deleted_feedback_items} feedback items.`)
    } catch (error) {
      setNotice(error instanceof APIRequestError ? error.message : 'Activity history could not be cleared.')
    } finally {
      setSaving(false)
    }
  }

  async function resetProfile() {
    if (!window.confirm('Reset the learned profile? Existing history will be kept, but it will no longer influence recommendations.')) return
    setSaving(true)
    try {
      await api.resetPersonalization(false)
      const refreshed = await api.personalizationProfile()
      setProfile(refreshed)
      setNotice('Learned personalization profile reset. New activity can shape it again.')
    } catch (error) {
      setNotice(error instanceof APIRequestError ? error.message : 'The profile could not be reset.')
    } finally {
      setSaving(false)
    }
  }

  function selectedRoots(event: ChangeEvent<HTMLSelectElement>) {
    return Array.from(event.currentTarget.selectedOptions, (option) => option.value)
  }

  if (loading) return <section aria-label="Personalization settings" className="mt-8 rounded-2xl border border-[#e3e8e4] bg-white p-5"><p role="status" className="text-sm text-[#89948d]">Loading personalization settings…</p></section>

  const current = preferences ?? defaultPreferences
  return <section aria-label="Personalization settings" className="mt-8 rounded-2xl border border-[#e3e8e4] bg-white p-5 sm:p-6">
    <div className="flex flex-wrap items-start justify-between gap-4"><div><p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#71917e]">Your control</p><h2 className="mt-1 text-lg font-semibold text-[#27352e]">Personalization</h2><p className="mt-1 max-w-2xl text-xs leading-5 text-[#89948d]">ScoutX learns from file opens and your direct feedback. Activity records omit search text and document contents; existing search-run records may retain the submitted query.</p></div><span className="rounded-full bg-[#eff4f0] px-3 py-1.5 text-xs capitalize text-[#537262]">{profile?.status ?? 'profile unavailable'}</span></div>
    {notice && <p role="status" className="mt-4 rounded-xl bg-[#f7f9f7] px-3 py-2 text-sm text-[#536158]">{notice}</p>}
    <div className="mt-5 grid gap-4 md:grid-cols-2">
      <div className="rounded-xl bg-[#f7f9f7] p-4"><label className="flex items-center justify-between gap-3 text-sm font-medium text-[#334139]">Use my activity to improve recommendations<input aria-label="Enable personalization" type="checkbox" checked={current.personalization_enabled} disabled={saving} onChange={(event) => void save({ ...current, personalization_enabled: event.target.checked })} className="h-4 w-4 accent-[#1c5b45]" /></label><p className="mt-2 text-xs leading-5 text-[#89948d]">When turned off, search uses the existing retrieval and reranking order. Your file library still works.</p><p className="mt-3 text-xs text-[#728177]">{profile?.recent_file_count ?? 0} recent files · {profile?.feedback_signal_count ?? 0} feedback signals</p></div>
      <div className="rounded-xl bg-[#f7f9f7] p-4"><label className="text-xs font-medium text-[#61736a]">Preferred file types<input aria-label="Preferred file types" value={preferredExtensions} onChange={(event) => setPreferredExtensions(event.target.value)} placeholder=".py, .pdf" className="mt-1 block w-full rounded-lg border border-[#e2e8e3] bg-white px-3 py-2 text-sm text-[#334139]" /></label><label className="mt-3 block text-xs font-medium text-[#61736a]">File types to exclude<input aria-label="File types to exclude" value={excludedExtensions} onChange={(event) => setExcludedExtensions(event.target.value)} placeholder=".tmp, .log" className="mt-1 block w-full rounded-lg border border-[#e2e8e3] bg-white px-3 py-2 text-sm text-[#334139]" /></label><button type="button" disabled={saving} onClick={() => void save({ ...current, preferred_extensions: preferredExtensions.split(',').map((value) => value.trim()).filter(Boolean), excluded_extensions: excludedExtensions.split(',').map((value) => value.trim()).filter(Boolean) })} className="mt-3 rounded-lg bg-[#1c5b45] px-3 py-2 text-xs font-semibold text-white disabled:opacity-50">Save file types</button></div>
      <label className="rounded-xl bg-[#f7f9f7] p-4 text-xs font-medium text-[#61736a]">Preferred folders<select aria-label="Preferred folders" multiple value={current.preferred_root_ids} onChange={(event) => void save({ ...current, preferred_root_ids: selectedRoots(event) })} className="mt-2 block h-24 w-full rounded-lg border border-[#e2e8e3] bg-white px-2 py-1 text-sm text-[#334139]">{roots.filter((root) => root.status === 'enabled').map((root) => <option key={root.id} value={root.id}>{root.display_name ?? root.path}</option>)}</select><span className="mt-1 block font-normal text-[#89948d]">Use Command-click or Ctrl-click to choose multiple folders.</span></label>
      <label className="rounded-xl bg-[#f7f9f7] p-4 text-xs font-medium text-[#61736a]">Folders to exclude<select aria-label="Folders to exclude" multiple value={current.excluded_root_ids} onChange={(event) => void save({ ...current, excluded_root_ids: selectedRoots(event) })} className="mt-2 block h-24 w-full rounded-lg border border-[#e2e8e3] bg-white px-2 py-1 text-sm text-[#334139]">{roots.map((root) => <option key={root.id} value={root.id}>{root.display_name ?? root.path}</option>)}</select><span className="mt-1 block font-normal text-[#89948d]">Excluded folders are omitted from personalized recommendations.</span></label>
    </div>
    <div className="mt-6 border-t border-[#edf0ee] pt-5"><div className="flex flex-wrap items-center justify-between gap-3"><div><h3 className="text-sm font-semibold text-[#334139]">Recent activity</h3><p className="mt-1 text-xs text-[#89948d]">Only file actions appear here. Search text and file contents are not shown.</p></div><div className="flex gap-2"><button type="button" disabled={saving} onClick={() => void resetProfile()} className="rounded-lg border border-[#e3e8e4] px-3 py-2 text-xs font-medium text-[#496153] disabled:opacity-50">Reset learned profile</button><button type="button" disabled={saving} onClick={() => void clearHistory()} className="rounded-lg border border-[#f0d7d2] px-3 py-2 text-xs font-medium text-[#965748] disabled:opacity-50">Clear history</button></div></div>
      {!activity.length ? <p className="mt-4 rounded-xl border border-dashed border-[#dce4de] p-4 text-sm text-[#89948d]">No recent activity has been recorded.</p> : <><ul className="mt-3 divide-y divide-[#edf0ee]">{activity.map((item) => <li key={item.id} className="flex items-center justify-between gap-4 py-3"><div className="min-w-0"><p className="truncate text-sm text-[#46534b]">{eventLabel(item.event_type)}{item.filename ? ` · ${item.filename}` : ''}</p><p className="mt-1 text-xs text-[#929c95]">{new Date(item.created_at).toLocaleString()}</p></div>{item.extension && <span className="shrink-0 text-xs text-[#829087]">{item.extension}</span>}</li>)}</ul><div className="mt-3 flex items-center justify-between text-xs text-[#839087]"><span>{activityOffset + 1}–{Math.min(activityOffset + activity.length, activityTotal)} of {activityTotal}</span><div className="flex gap-2"><button type="button" disabled={activityOffset === 0 || saving} onClick={() => void loadActivity(Math.max(0, activityOffset - 12))} className="rounded-lg border border-[#e2e8e3] px-3 py-1.5 disabled:opacity-40">Newer</button><button type="button" disabled={activityOffset + 12 >= activityTotal || saving} onClick={() => void loadActivity(activityOffset + 12)} className="rounded-lg border border-[#e2e8e3] px-3 py-1.5 disabled:opacity-40">Older</button></div></div></>}</div>
  </section>
}
