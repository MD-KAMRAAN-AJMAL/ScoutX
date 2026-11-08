import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { FileList } from './FileList'
import type { FileRecord } from '../services/api'

const file: FileRecord = {
  id: 'file-1', root_id: 'root-1', name: 'paper.pdf', path: 'Research/paper.pdf',
  mime_type: 'application/pdf', extension: '.pdf', size_bytes: 2048,
  indexing_status: 'indexed', version: 1,
}

describe('FileList', () => {
  it('renders safe file metadata and sends selection by stable ID', async () => {
    const onSelect = vi.fn()
    const user = userEvent.setup()
    render(<FileList files={[file]} onSelect={onSelect} />)
    await user.click(screen.getByRole('button', { name: 'paper.pdf' }))
    expect(screen.getByText('Research/paper.pdf')).toBeInTheDocument()
    expect(onSelect).toHaveBeenCalledWith(file)
  })
})
