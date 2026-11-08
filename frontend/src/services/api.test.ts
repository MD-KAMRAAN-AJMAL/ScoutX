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
})
