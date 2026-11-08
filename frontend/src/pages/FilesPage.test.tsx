import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { FilesPage } from './FilesPage'
import { api, type FileDetailResponse, type FileListResponse, type IndexStatus } from '../services/api'

vi.mock('../services/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../services/api')>()
  return {
    ...actual,
    api: {
      ...actual.api,
      files: vi.fn(),
      roots: vi.fn(),
      indexStatus: vi.fn(),
      fileDetail: vi.fn(),
      feedback: vi.fn(),
    },
  }
})

const fileList: FileListResponse = {
  items: [{
    id: 'file-42', root_id: 'root-1', name: 'budget.xlsx', path: 'Finance/budget.xlsx',
    mime_type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    extension: '.xlsx', size_bytes: 4096, modified_at: '2026-11-08T00:00:00Z',
    indexing_status: 'indexed', version: 1,
  }],
  limit: 50, offset: 0, total: 1,
}
const status: IndexStatus = {
  state: 'idle', files_discovered: 1, files_indexed: 1, files_failed: 0,
  status_counts: { pending: 0, processing: 0, indexed: 1, partial: 0, failed: 0, unsupported: 0, deleted: 0 },
  roots_registered: 1,
}
const fileDetail: FileDetailResponse = { file: fileList.items[0], snippets: [] }

describe('FilesPage', () => {
  beforeEach(() => {
    vi.mocked(api.roots).mockResolvedValue({ items: [{ id: 'root-1', path: 'Documents', status: 'enabled' }] })
    vi.mocked(api.files).mockResolvedValue(fileList)
    vi.mocked(api.indexStatus).mockResolvedValue(status)
    vi.mocked(api.fileDetail).mockResolvedValue(fileDetail)
    vi.mocked(api.feedback).mockResolvedValue({ accepted: true })
  })

  it('filters through the server and opens details by file ID', async () => {
    const user = userEvent.setup()
    render(<FilesPage />)
    await screen.findByRole('button', { name: 'budget.xlsx' })
    await user.type(screen.getByLabelText('Filename'), 'budget')
    await user.click(screen.getByRole('button', { name: 'Filter' }))
    await waitFor(() => expect(api.files).toHaveBeenLastCalledWith(expect.objectContaining({ filenameContains: 'budget' })) )
    await user.click(screen.getByRole('button', { name: 'budget.xlsx' }))
    expect(await screen.findByRole('complementary', { name: 'File details' })).toBeInTheDocument()
    expect(api.fileDetail).toHaveBeenCalledWith('file-42', expect.any(AbortSignal))
  })
})
