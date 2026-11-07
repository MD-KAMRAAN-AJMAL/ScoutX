const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000/api/v1').replace(/\/$/, '')

export interface APIError {
  code: string
  message: string
  details?: unknown
}

export interface SearchRequest {
  query: string
  limit?: number
  mode?: 'auto' | 'keyword' | 'semantic' | 'hybrid'
}

export interface SearchResponse {
  query: string
  results: Array<{
    file_id: string
    name: string
    path: string
    score: number
    excerpt?: string | null
    source_reference?: string | null
  }>
  answer?: string | null
}

export interface FileRecord {
  id: string
  root_id: string
  name: string
  path: string
  mime_type?: string | null
  size_bytes: number
  modified_at?: string | null
  indexing_status: 'pending' | 'processing' | 'indexed' | 'partial' | 'failed' | 'unsupported' | 'deleted'
  extraction_error?: string | null
}

export interface FileListResponse {
  items: FileRecord[]
  limit: number
  offset: number
  total: number
}

export interface RootRecord {
  id: string
  path: string
  display_name?: string | null
  status: 'enabled' | 'disabled'
  last_reconciled_at?: string | null
}

export interface IndexStatus {
  state: string
  files_discovered: number
  files_indexed: number
  files_failed: number
  status_counts: Record<FileRecord['indexing_status'], number>
  roots_registered: number
  last_reconciled_at?: string | null
}

export interface ReconcileResponse {
  accepted: boolean
  root_ids: string[]
  task_ids: Array<string | null>
  submitted_at: string
}

export class APIRequestError extends Error {
  readonly status: number
  readonly code: string
  readonly details?: unknown

  constructor(status: number, error: APIError) {
    super(error.message)
    this.name = 'APIRequestError'
    this.status = status
    this.code = error.code
    this.details = error.details
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...init?.headers },
    })
  } catch {
    throw new APIRequestError(0, {
      code: 'api_unavailable',
      message: 'Could not reach the ScoutX API. Check that the backend is running.',
    })
  }

  const payload: unknown = response.status === 204 ? undefined : await response.json().catch(() => undefined)
  if (!response.ok) {
    const error = (payload as { error?: APIError } | undefined)?.error
    throw new APIRequestError(response.status, error ?? {
      code: 'http_error',
      message: `The API returned status ${response.status}.`,
    })
  }
  return payload as T
}

export const api = {
  health: () => request<{ status: string; service: string }>('/health'),
  search: (body: SearchRequest) => request<SearchResponse>('/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }),
  files: (limit = 50, offset = 0, rootId?: string) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
    if (rootId) params.set('root_id', rootId)
    return request<FileListResponse>(`/files?${params.toString()}`)
  },
  roots: () => request<{ items: RootRecord[] }>('/roots'),
  addRoot: (path: string, displayName?: string) => request<RootRecord>('/roots', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, display_name: displayName }),
  }),
  removeRoot: (id: string) => request<void>(`/roots/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  indexStatus: () => request<IndexStatus>('/index/status'),
  reconcile: (rootId?: string) => request<ReconcileResponse>('/index/reconcile', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(rootId ? { root_id: rootId } : {}),
  }),
  feedback: (fileId: string, action: 'opened' | 'helpful' | 'not_helpful' | 'dismissed', query?: string) =>
    request<{ accepted: boolean }>('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_id: fileId, action, query }),
    }),
}
