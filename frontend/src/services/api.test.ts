import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, APIRequestError } from './api'

afterEach(() => vi.unstubAllGlobals())

describe('API client', () => {
  it('converts structured backend errors into APIRequestError', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      error: { code: 'file_inaccessible', message: 'ScoutX cannot currently read this file.' },
    }), { status: 403, headers: { 'Content-Type': 'application/json' } })))

    await expect(api.fileDetail('file-id')).rejects.toMatchObject<Partial<APIRequestError>>({
      name: 'APIRequestError', status: 403, code: 'file_inaccessible',
    })
  })

  it('propagates an aborted request so screens can ignore stale responses', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new DOMException('aborted', 'AbortError')))
    const controller = new AbortController()
    controller.abort()

    await expect(api.health(controller.signal)).rejects.toMatchObject({ name: 'AbortError' })
  })

  it('uses the personalized recommendation and preference API contracts', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ items: [], category: 'related', limit: 5 }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    await api.recommendations('related', 5, '.py')
    expect(fetchMock.mock.calls[0][0]).toContain('/recommendations?category=related&limit=5&extension=.py')

    const preferences = {
      personalization_enabled: false, preferred_extensions: ['.py'], excluded_extensions: [],
      preferred_root_ids: [], excluded_root_ids: [], recommendation_limit: 12,
    }
    await api.updatePersonalizationPreferences(preferences)
    expect(fetchMock.mock.calls[1][1]).toMatchObject({ method: 'PUT' })
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual(preferences)
  })
})
