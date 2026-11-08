import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { SettingsPage } from './SettingsPage'
import { api, type IndexStatus } from '../services/api'

vi.mock('../services/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../services/api')>()
  return {
    ...actual,
    api: {
      ...actual.api,
      roots: vi.fn(),
      indexStatus: vi.fn(),
      removeRoot: vi.fn(),
      personalizationProfile: vi.fn(),
      personalizationPreferences: vi.fn(),
      activity: vi.fn(),
    },
  }
})

vi.mock('../components/PersonalizationSettings', () => ({
  PersonalizationSettings: () => null,
}))

const status: IndexStatus = {
  state: 'idle',
  files_discovered: 0,
  files_indexed: 0,
  files_failed: 0,
  status_counts: { pending: 0, processing: 0, indexed: 0, partial: 0, failed: 0, unsupported: 0, deleted: 0 },
  roots_registered: 1,
}

describe('SettingsPage', () => {
  beforeEach(() => {
    vi.mocked(api.roots).mockResolvedValue({ items: [{ id: 'root-1', path: 'Demo files', status: 'enabled' }] })
    vi.mocked(api.indexStatus).mockResolvedValue(status)
    vi.mocked(api.removeRoot).mockResolvedValue(undefined)
  })

  it('disables a folder and explains that its indexed records remain', async () => {
    const user = userEvent.setup()
    render(<SettingsPage />)
    await screen.findByText('Monitoring enabled')

    await user.click(screen.getByRole('button', { name: 'Disable folder' }))

    await waitFor(() => expect(api.removeRoot).toHaveBeenCalledWith('root-1'))
    expect(await screen.findByText('Monitoring disabled')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('Existing indexed records have been kept.')
  })
})
