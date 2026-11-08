const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000/api/v1').replace(/\/$/, '')

export interface APIError {
  code: string
  message: string
  details?: unknown
}

export type FileStatus = 'pending' | 'processing' | 'indexed' | 'partial' | 'failed' | 'unsupported' | 'deleted'
export type SearchMode = 'auto' | 'keyword' | 'semantic' | 'hybrid'
export type FeedbackAction = 'opened' | 'selected' | 'relevant' | 'not_relevant' | 'helpful' | 'not_helpful' | 'useful' | 'not_useful' | 'dismissed'
export type RecommendationCategory = 'recommended_for_you' | 'recently_accessed' | 'continue_working' | 'frequently_used' | 'related' | 'recently_indexed'

export interface SearchRequest {
  query: string
  limit?: number
  mode?: SearchMode
  generate_answer?: boolean
  conversation_id?: string
  extensions?: string[]
  root_id?: string
  modified_after?: string
  modified_before?: string
}

export interface SearchResult {
  file_id: string
  chunk_id: string | null
  name: string
  path: string
  score: number
  excerpt?: string | null
  source_reference?: string | null
  modality: string
  extension?: string | null
  retrieval_method: string
}

export interface SupportingChunk {
  chunk_id: string | null
  content: string
  snippet: string
  score: number
  retrieval_score: number
  retrieval_method: string
  content_version?: number | null
  section_title?: string | null
  source_references: Array<Record<string, unknown>>
  page_number?: number | null
  line_start?: number | null
  line_end?: number | null
  start_time_ms?: number | null
  end_time_ms?: number | null
}

export interface FileRecommendation {
  file_id: string
  file_name: string
  file_path: string
  extension?: string | null
  mime_type?: string | null
  score: number
  scoring_method: string
  retrieval_sources: string[]
  personalization_reasons?: string[]
  supporting_chunks: SupportingChunk[]
}

export interface CitationLocation {
  citation_id: string
  chunk_id: string | null
  content_version?: number | null
  relevance_score: number
  section_title?: string | null
  source_references: Array<Record<string, unknown>>
  page_number?: number | null
  line_start?: number | null
  line_end?: number | null
  start_time_ms?: number | null
  end_time_ms?: number | null
}

export interface AnswerCitation {
  citation_id: string
  citation_ids: string[]
  file_id: string
  chunk_id: string | null
  file_name: string
  file_path: string
  content_version?: number | null
  relevance_score: number
  section_title?: string | null
  source_references: Array<Record<string, unknown>>
  page_number?: number | null
  line_start?: number | null
  line_end?: number | null
  start_time_ms?: number | null
  end_time_ms?: number | null
  locations: CitationLocation[]
}

export interface SearchResponse {
  query: string
  response_mode: 'retrieval_only' | 'answer'
  results: SearchResult[]
  files: FileRecommendation[]
  citations: AnswerCitation[]
  answer?: string | null
  run_id: string
  conversation_id: string
  intent: Record<string, unknown>
  retrieval_plan: Record<string, unknown>
  tools_invoked: string[]
  candidate_counts: Record<string, number>
  retry_count: number
  execution_trace: string[]
  warnings: string[]
  reranking_status: string
  generation_status: string
  personalization_status: string
  latency_ms: Record<string, number>
}

export interface RecommendationItem {
  id: string
  file_id: string
  filename: string
  path: string
  extension?: string | null
  mime_type?: string | null
  modified_at?: string | null
  score: number
  category: RecommendationCategory
  reason: string
}

export interface RecommendationListResponse {
  items: RecommendationItem[]
  category: RecommendationCategory
  limit: number
}

export interface PersonalizationPreferences {
  personalization_enabled: boolean
  preferred_extensions: string[]
  excluded_extensions: string[]
  preferred_root_ids: string[]
  excluded_root_ids: string[]
  recommendation_limit: number
}

export interface PersonalizationProfile {
  status: 'available' | 'cold_start' | 'disabled'
  personalization_enabled: boolean
  preferred_extensions: string[]
  frequent_roots: Array<{ id: string; name: string }>
  recent_file_count: number
  feedback_signal_count: number
}

export interface ActivityItem {
  id: string
  event_type: string
  file_id?: string | null
  filename?: string | null
  extension?: string | null
  created_at: string
}

export interface ActivityListResponse {
  items: ActivityItem[]
  limit: number
  offset: number
  total: number
}

export interface FileRecord {
  id: string
  root_id: string
  name: string
  path: string
  mime_type?: string | null
  extension?: string | null
  size_bytes: number
  modified_at?: string | null
  indexing_status: FileStatus
  parser_status?: string | null
  extraction_warnings?: unknown[]
  extraction_error?: string | null
  version: number
  last_indexed_at?: string | null
  created_at?: string | null
}

export interface FileSnippet {
  chunk_id: string
  snippet: string
  modality: string
  content_version: number
  section_title?: string | null
  page_number?: number | null
  line_start?: number | null
  line_end?: number | null
  start_time_ms?: number | null
  end_time_ms?: number | null
}

export interface FileDetailResponse {
  file: FileRecord
  snippets: FileSnippet[]
}

export interface FileListOptions {
  limit?: number
  offset?: number
  rootId?: string
  filenameContains?: string
  extension?: string
  indexingStatus?: FileStatus
  modifiedAfter?: string
  modifiedBefore?: string
  sortBy?: 'name' | 'modified_at' | 'size_bytes' | 'last_indexed_at'
  sortOrder?: 'asc' | 'desc'
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
  status_counts: Record<FileStatus, number>
  roots_registered: number
  last_reconciled_at?: string | null
  recent_jobs?: IndexJobProgress[]
  recent_job_counts?: Record<'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled', number>
}

export interface IndexJobProgress {
  id: string
  job_type: string
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
  root_id?: string | null
  file_id?: string | null
  attempts: number
  progress_percent?: number | null
  stage?: string | null
  error_message?: string | null
  created_at: string
  started_at?: string | null
  finished_at?: string | null
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  service: string
  dependencies: Record<'postgres' | 'qdrant' | 'redis' | 'ollama', 'available' | 'unavailable'>
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

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...init.headers },
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
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
  health: (signal?: AbortSignal) => request<HealthResponse>('/health', { signal }),
  search: (body: SearchRequest, signal?: AbortSignal) => request<SearchResponse>('/search', {
    method: 'POST',
    signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }),
  files: (options: FileListOptions = {}, signal?: AbortSignal) => {
    const params = new URLSearchParams({ limit: String(options.limit ?? 50), offset: String(options.offset ?? 0) })
    if (options.rootId) params.set('root_id', options.rootId)
    if (options.filenameContains) params.set('filename_contains', options.filenameContains)
    if (options.extension) {
      options.extension.split(',').map((item) => item.trim()).filter(Boolean)
        .forEach((item) => params.append('extension', item))
    }
    if (options.indexingStatus) params.set('indexing_status', options.indexingStatus)
    if (options.modifiedAfter) params.set('modified_after', options.modifiedAfter)
    if (options.modifiedBefore) params.set('modified_before', options.modifiedBefore)
    if (options.sortBy) params.set('sort_by', options.sortBy)
    if (options.sortOrder) params.set('sort_order', options.sortOrder)
    return request<FileListResponse>(`/files?${params.toString()}`, { signal })
  },
  fileDetail: (fileId: string, signal?: AbortSignal) =>
    request<FileDetailResponse>(`/files/${encodeURIComponent(fileId)}`, { signal }),
  roots: (signal?: AbortSignal) => request<{ items: RootRecord[] }>('/roots', { signal }),
  addRoot: (path: string, displayName?: string) => request<RootRecord>('/roots', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, display_name: displayName }),
  }),
  removeRoot: (id: string) => request<void>(`/roots/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  indexStatus: (signal?: AbortSignal) => request<IndexStatus>('/indexing/status', { signal }),
  reconcile: (rootId?: string) => request<ReconcileResponse>('/index/reconcile', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(rootId ? { root_id: rootId } : {}),
  }),
  feedback: (fileId: string, action: FeedbackAction, query?: string, agentRunId?: string, signal?: AbortSignal) =>
    request<{ accepted: boolean }>('/feedback', {
      method: 'POST',
      signal,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_id: fileId, action, query, agent_run_id: agentRunId }),
    }),
  recommendations: (category: RecommendationCategory = 'recommended_for_you', limit = 12, extension?: string, signal?: AbortSignal) => {
    const params = new URLSearchParams({ category, limit: String(limit) })
    if (extension) params.set('extension', extension)
    return request<RecommendationListResponse>(`/recommendations?${params.toString()}`, { signal })
  },
  personalizationProfile: (signal?: AbortSignal) => request<PersonalizationProfile>('/personalization/profile', { signal }),
  personalizationPreferences: (signal?: AbortSignal) => request<PersonalizationPreferences>('/personalization/preferences', { signal }),
  updatePersonalizationPreferences: (preferences: PersonalizationPreferences, signal?: AbortSignal) => request<PersonalizationPreferences>('/personalization/preferences', {
    method: 'PUT', signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(preferences),
  }),
  activity: (options: { limit?: number; offset?: number } = {}, signal?: AbortSignal) => {
    const params = new URLSearchParams({ limit: String(options.limit ?? 20), offset: String(options.offset ?? 0) })
    return request<ActivityListResponse>(`/activity?${params.toString()}`, { signal })
  },
  clearActivity: (signal?: AbortSignal) => request<{ deleted_activity_events: number; deleted_feedback_items: number }>('/activity', { method: 'DELETE', signal }),
  resetPersonalization: (clearActivityHistory = false, signal?: AbortSignal) => request<{ reset: boolean; activity_history_cleared: boolean }>('/personalization/reset', {
    method: 'POST', signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ clear_activity_history: clearActivityHistory }),
  }),
}
