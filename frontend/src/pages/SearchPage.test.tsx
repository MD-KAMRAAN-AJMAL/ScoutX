import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { SearchPage } from './SearchPage'
import { api, type FileDetailResponse, type SearchResponse } from '../services/api'

vi.mock('../services/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../services/api')>()
  return {
    ...actual,
    api: {
      ...actual.api,
      search: vi.fn(),
      fileDetail: vi.fn(),
      feedback: vi.fn(),
      roots: vi.fn(),
      recommendations: vi.fn(),
    },
  }
})

const response: SearchResponse = {
  query: 'where is the total?',
  response_mode: 'answer',
  results: [],
  files: [{
    file_id: 'file-1',
    file_name: 'budget.xlsx',
    file_path: 'Finance/budget.xlsx',
    extension: '.xlsx',
    mime_type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    score: 0.91,
    scoring_method: 'best_chunk_plus_capped_support',
    retrieval_sources: ['semantic'],
    supporting_chunks: [{
      chunk_id: 'chunk-1',
      content: 'The yearly total is 4200.',
      snippet: 'The yearly total is 4200.',
      score: 0.91,
      retrieval_score: 0.7,
      retrieval_method: 'semantic',
      content_version: 2,
      section_title: 'Summary',
      source_references: [],
      page_number: 3,
    }],
  }],
  citations: [{
    citation_id: 'S1',
    citation_ids: ['S1'],
    file_id: 'file-1',
    chunk_id: 'chunk-1',
    file_name: 'budget.xlsx',
    file_path: 'Finance/budget.xlsx',
    content_version: 2,
    relevance_score: 0.91,
    source_references: [],
    locations: [{
      citation_id: 'S1',
      chunk_id: 'chunk-1',
      content_version: 2,
      relevance_score: 0.91,
      source_references: [],
      page_number: 3,
    }],
  }],
  answer: 'The yearly total is 4200 [S1].',
  run_id: 'run-1',
  conversation_id: 'conversation-1',
  intent: {},
  retrieval_plan: {},
  tools_invoked: ['semantic'],
  candidate_counts: { semantic: 1 },
  retry_count: 0,
  execution_trace: [],
  warnings: [],
  reranking_status: 'qwen3',
  generation_status: 'succeeded',
    personalization_status: 'cold_start',
  latency_ms: { retrieval: 12, reranking: 4, generation: 20 },
}

const detail: FileDetailResponse = {
  file: {
    id: 'file-1', root_id: 'root-1', name: 'budget.xlsx', path: 'Finance/budget.xlsx',
    mime_type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    extension: '.xlsx', size_bytes: 1024, modified_at: null, indexing_status: 'indexed',
    version: 2, last_indexed_at: null,
  },
  snippets: [{
    chunk_id: 'chunk-1', snippet: 'The yearly total is 4200.', modality: 'text',
    content_version: 2, section_title: 'Summary', page_number: 3,
  }],
}

describe('SearchPage', () => {
  beforeEach(() => {
    vi.mocked(api.feedback).mockResolvedValue({ accepted: true })
    vi.mocked(api.fileDetail).mockResolvedValue(detail)
    vi.mocked(api.roots).mockResolvedValue({ items: [] })
    vi.mocked(api.recommendations).mockResolvedValue({ items: [], category: 'recommended_for_you', limit: 8 })
  })

  it('submits a natural-language query and renders ranked files and answer', async () => {
    vi.mocked(api.search).mockResolvedValue(response)
    const user = userEvent.setup()
    render(<SearchPage />)

    await user.type(screen.getByRole('textbox', { name: 'Search your files' }), 'where is the total?{Enter}')

    expect(await screen.findByText('The yearly total is 4200')).toBeInTheDocument()
    expect(screen.getByText('budget.xlsx')).toBeInTheDocument()
    expect(api.search).toHaveBeenCalledWith(
      expect.objectContaining({ query: 'where is the total?', generate_answer: true, mode: 'auto' }),
      expect.any(AbortSignal),
    )
  })

  it('opens a source detail when an answer citation is selected', async () => {
    vi.mocked(api.search).mockResolvedValue(response)
    const user = userEvent.setup()
    render(<SearchPage />)
    await user.type(screen.getByRole('textbox', { name: 'Search your files' }), 'where is the total?{Enter}')
    await user.click(await screen.findByRole('button', { name: 'Open citation [S1]' }))

    const panel = await screen.findByRole('complementary', { name: 'File details' })
    expect(within(panel).getByText('The yearly total is 4200.')).toBeInTheDocument()
    expect(api.fileDetail).toHaveBeenCalledWith('file-1', expect.any(AbortSignal))
  })

  it('shows the no-results response without inventing result cards', async () => {
    vi.mocked(api.search).mockResolvedValue({
      ...response, answer: "I couldn't find indexed evidence that answers this request.",
      files: [], results: [], citations: [], generation_status: 'no_results',
    })
    const user = userEvent.setup()
    render(<SearchPage />)
    await user.type(screen.getByRole('textbox', { name: 'Search your files' }), 'unknown{Enter}')

    expect(await screen.findByText('No matching files found')).toBeInTheDocument()
    expect(screen.queryByText('budget.xlsx')).not.toBeInTheDocument()
  })

  it('surfaces safe API errors and cancels an in-flight request', async () => {
    vi.mocked(api.search).mockRejectedValueOnce(new Error('connection refused'))
    const user = userEvent.setup()
    render(<SearchPage />)
    const input = screen.getByRole('textbox', { name: 'Search your files' })
    await user.type(input, 'failure{Enter}')
    expect(await screen.findByRole('alert')).toHaveTextContent('Search could not be completed.')

    let signal: AbortSignal | undefined
    vi.mocked(api.search).mockImplementationOnce((_body, nextSignal) => {
      signal = nextSignal
      return new Promise(() => undefined)
    })
    await user.clear(input)
    await user.type(input, 'cancel me{Enter}')
    await user.click(await screen.findByRole('button', { name: 'Cancel' }))
    expect(signal?.aborted).toBe(true)
  })
})
