import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { RecommendationsPanel } from './RecommendationsPanel'
import { api } from '../services/api'

vi.mock('../services/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../services/api')>()
  return { ...actual, api: { ...actual.api, recommendations: vi.fn(), feedback: vi.fn() } }
})

const item = {
  id: 'file-1', file_id: 'file-1', filename: 'analysis.py', path: 'project/analysis.py',
  extension: '.py', mime_type: 'text/x-python', modified_at: null, score: 0.7,
  category: 'recommended_for_you' as const, reason: 'Matches a file type you prefer',
}

describe('RecommendationsPanel', () => {
  beforeEach(() => {
    vi.mocked(api.recommendations).mockReset()
    vi.mocked(api.feedback).mockReset().mockResolvedValue({ accepted: true })
  })

  it('renders concise recommendation explanations and supports dismissal', async () => {
    vi.mocked(api.recommendations).mockResolvedValue({ items: [item], category: 'recommended_for_you', limit: 8 })
    render(<RecommendationsPanel />)
    expect(await screen.findByText('analysis.py')).toBeInTheDocument()
    expect(screen.getByText('Matches a file type you prefer')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }))
    await waitFor(() => expect(screen.queryByText('analysis.py')).not.toBeInTheDocument())
    expect(api.feedback).toHaveBeenCalledWith('file-1', 'dismissed')
  })

  it('shows a safe message when recommendation loading fails', async () => {
    vi.mocked(api.recommendations).mockRejectedValue(new Error('offline'))
    render(<RecommendationsPanel />)
    expect(await screen.findByText('Recommendations are unavailable.')).toBeInTheDocument()
  })
})
