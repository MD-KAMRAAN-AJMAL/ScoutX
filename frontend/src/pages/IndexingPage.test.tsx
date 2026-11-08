import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { IndexingPage } from './IndexingPage'
import { api, type IndexStatus } from '../services/api'

vi.mock('../services/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../services/api')>()
  return { ...actual, api: { ...actual.api, indexStatus: vi.fn() } }
})

const status: IndexStatus = {
  state: 'processing', files_discovered: 20, files_indexed: 14, files_failed: 1,
  status_counts: { pending: 2, processing: 3, indexed: 14, partial: 0, failed: 1, unsupported: 0, deleted: 0 },
  roots_registered: 2, last_reconciled_at: '2026-11-08T08:00:00Z',
  recent_jobs: [], recent_job_counts: { queued: 2, running: 3, succeeded: 14, failed: 1, cancelled: 0 },
}

describe('IndexingPage', () => {
  beforeEach(() => vi.mocked(api.indexStatus).mockResolvedValue(status))

  it('shows discovered, indexed, failed, and reconciliation status', async () => {
    render(<IndexingPage />)
    expect(await screen.findByText('14 indexed')).toBeInTheDocument()
    expect(screen.getByText('20 discovered')).toBeInTheDocument()
    expect(screen.getByText(/Last directory reconciliation/)).toBeInTheDocument()
    expect(api.indexStatus).toHaveBeenCalledWith(expect.any(AbortSignal))
  })
})
