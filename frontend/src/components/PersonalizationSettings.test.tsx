import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { PersonalizationSettings } from './PersonalizationSettings'
import { api, type PersonalizationPreferences } from '../services/api'

vi.mock('../services/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../services/api')>()
  return { ...actual, api: {
    ...actual.api,
    personalizationPreferences: vi.fn(), personalizationProfile: vi.fn(), activity: vi.fn(),
    updatePersonalizationPreferences: vi.fn(), clearActivity: vi.fn(), resetPersonalization: vi.fn(),
  } }
})

const preferences: PersonalizationPreferences = {
  personalization_enabled: true, preferred_extensions: ['.py'], excluded_extensions: [],
  preferred_root_ids: [], excluded_root_ids: [], recommendation_limit: 12,
}

describe('PersonalizationSettings', () => {
  beforeEach(() => {
    vi.mocked(api.personalizationPreferences).mockResolvedValue(preferences)
    vi.mocked(api.personalizationProfile).mockResolvedValue({ status: 'available', personalization_enabled: true, preferred_extensions: ['.py'], frequent_roots: [], recent_file_count: 2, feedback_signal_count: 1 })
    vi.mocked(api.activity).mockResolvedValue({ items: [], limit: 12, offset: 0, total: 0 })
    vi.mocked(api.updatePersonalizationPreferences).mockImplementation(async (value) => value)
    vi.mocked(api.clearActivity).mockResolvedValue({ deleted_activity_events: 2, deleted_feedback_items: 1 })
    vi.mocked(api.resetPersonalization).mockResolvedValue({ reset: true, activity_history_cleared: false })
  })

  it('saves the enabled/disabled setting', async () => {
    render(<PersonalizationSettings roots={[]} />)
    const toggle = await screen.findByRole('checkbox', { name: 'Enable personalization' })
    fireEvent.click(toggle)
    await waitFor(() => expect(api.updatePersonalizationPreferences).toHaveBeenCalledWith({ ...preferences, personalization_enabled: false }))
  })

  it('asks before clearing activity and feedback history', async () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
    render(<PersonalizationSettings roots={[]} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Clear history' }))
    expect(confirm).toHaveBeenCalled()
    expect(api.clearActivity).not.toHaveBeenCalled()
    confirm.mockReturnValue(true)
    fireEvent.click(screen.getByRole('button', { name: 'Clear history' }))
    await waitFor(() => expect(api.clearActivity).toHaveBeenCalledOnce())
    confirm.mockRestore()
  })

  it('resets the learned profile without deleting history', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    render(<PersonalizationSettings roots={[]} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Reset learned profile' }))
    await waitFor(() => expect(api.resetPersonalization).toHaveBeenCalledWith(false))
    expect(api.clearActivity).not.toHaveBeenCalled()
    vi.restoreAllMocks()
  })

  it('pages through recent activity', async () => {
    const rows = Array.from({ length: 12 }, (_, index) => ({
      id: `event-${index}`, event_type: 'opened', file_id: `file-${index}`,
      filename: `file-${index}.py`, extension: '.py', created_at: '2026-11-08T10:00:00Z',
    }))
    vi.mocked(api.activity).mockResolvedValue({ items: rows, limit: 12, offset: 0, total: 20 })
    const user = userEvent.setup()
    render(<PersonalizationSettings roots={[]} />)
    await screen.findByText('1–12 of 20')
    await user.click(screen.getByRole('button', { name: 'Older' }))
    await waitFor(() => expect(api.activity).toHaveBeenLastCalledWith({ limit: 12, offset: 12 }))
    expect(await screen.findByText('13–20 of 20')).toBeInTheDocument()
  })
})
