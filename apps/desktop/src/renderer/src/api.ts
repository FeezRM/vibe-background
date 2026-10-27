export type BackgroundMode = 'preset' | 'file' | 'blur_original'
export type LightingMode = 'off' | 'enhance' | 'match' | 'relight'
export type Tier = 'fast' | 'balanced' | 'max'

export interface Settings {
  background: {
    mode: BackgroundMode
    preset?: string
    path?: string
    dof: number
    blur: number
  }
  tier: Tier
  codec: 'h264' | 'hevc'
  placement: { scale: number; x: number; y: number }
  lighting: {
    mode: LightingMode
    strength: number
    wrap: number
    shadow: number
    warmth: number
    denoise: boolean
  }
}

export interface VideoInfo {
  width: number
  height: number
  fps: number
  duration: number
  frames: number
  has_audio: boolean
  is_hdr: boolean
}

export interface PresetInfo {
  id: string
  name: string
  type: 'image' | 'video'
  builtin: boolean
  tags: string[]
  subject: { scale: number; x: number; y: number }
  light: { shadow: number; warmth: number }
  license: string
  source: string
  url: string | null
}

export interface Job {
  id: string
  src: string
  out: string
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled'
  stage: 'relight' | 'render'
  frame: number
  total: number
  elapsed: number
  eta: number | null
  error: string | null
}

export class EngineError extends Error {}

export class Engine {
  constructor(
    readonly url: string,
    private readonly token: string
  ) {}

  /** URL usable directly in <img>/<video> src attributes. */
  mediaUrl(path: string, query: Record<string, string> = {}): string {
    const params = new URLSearchParams({ ...query, token: this.token })
    return `${this.url}${path}?${params}`
  }

  private async request(path: string, init: RequestInit = {}): Promise<Response> {
    const res = await fetch(`${this.url}${path}`, {
      ...init,
      headers: { 'content-type': 'application/json', 'x-vibebg-token': this.token, ...init.headers }
    })
    if (!res.ok) {
      let detail = res.statusText
      try {
        const body = await res.json()
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      } catch {
        // not JSON; keep the status text
      }
      throw new EngineError(detail)
    }
    return res
  }

  private async json<T>(path: string, body?: unknown, method = body ? 'POST' : 'GET'): Promise<T> {
    const res = await this.request(path, {
      method,
      body: body === undefined ? undefined : JSON.stringify(body)
    })
    return res.json()
  }

  health(): Promise<{ ok: boolean; version: string; device: string }> {
    return this.json('/health')
  }

  probe(path: string): Promise<VideoInfo> {
    return this.json('/probe', { path })
  }

  async frame(path: string, time: number, signal?: AbortSignal): Promise<Blob> {
    const res = await this.request('/frame', {
      method: 'POST',
      body: JSON.stringify({ path, time }),
      signal
    })
    return res.blob()
  }

  async preview(
    path: string,
    time: number,
    settings: Settings,
    signal?: AbortSignal
  ): Promise<Blob> {
    const res = await this.request('/preview', {
      method: 'POST',
      body: JSON.stringify({ path, time, settings }),
      signal
    })
    return res.blob()
  }

  submitJob(body: {
    path: string
    out?: string
    settings: Settings
    start?: number
    duration?: number
  }): Promise<Job> {
    return this.json('/jobs', body)
  }

  job(id: string): Promise<Job> {
    return this.json(`/jobs/${id}`)
  }

  cancelJob(id: string): Promise<Job> {
    return this.json(`/jobs/${id}/cancel`, {})
  }

  presets(): Promise<PresetInfo[]> {
    return this.json('/presets')
  }

  addPreset(path: string, name: string): Promise<{ id: string }> {
    return this.json('/presets', { path, name })
  }

  deletePreset(id: string): Promise<{ ok: boolean }> {
    return this.json(`/presets/${id}`, undefined, 'DELETE')
  }
}

/** Polls a job until it finishes, reporting progress along the way. */
export async function waitForJob(
  engine: Engine,
  id: string,
  onUpdate: (job: Job) => void,
  intervalMs = 400
): Promise<Job> {
  for (;;) {
    const job = await engine.job(id)
    onUpdate(job)
    if (job.status === 'done' || job.status === 'failed' || job.status === 'cancelled') return job
    await new Promise((r) => setTimeout(r, intervalMs))
  }
}

export const DEFAULT_SETTINGS: Settings = {
  background: { mode: 'blur_original', dof: 0.4, blur: 0.5 },
  tier: 'fast',
  codec: 'h264',
  placement: { scale: 1, x: 0, y: 0 },
  lighting: { mode: 'enhance', strength: 0.6, wrap: 0.5, shadow: 0, warmth: 0, denoise: false }
}
