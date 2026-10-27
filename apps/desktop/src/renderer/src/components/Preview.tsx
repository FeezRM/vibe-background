import {
  useEffect,
  useRef,
  useState,
  type PointerEvent,
  type ReactNode,
  type WheelEvent
} from 'react'
import { waitForJob, type Engine, type Settings, type VideoInfo } from '../api'
import { formatTime } from '../format'

const DEBOUNCE_MS = 120
const CLIP_SECONDS = 5

type Placement = Settings['placement']

export function Preview({
  engine,
  path,
  info,
  settings,
  canFrame,
  onPlacement
}: {
  engine: Engine
  path: string
  info: VideoInfo
  settings: Settings
  canFrame: boolean
  onPlacement: (p: Placement) => void
}): ReactNode {
  const [time, setTime] = useState(Math.min(1, info.duration / 2))
  const [image, setImage] = useState<string | null>(null)
  const [original, setOriginal] = useState<string | null>(null)
  const [comparing, setComparing] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [clip, setClip] = useState<
    | { state: 'rendering'; progress: number; relighting: boolean }
    | { state: 'ready'; url: string }
    | null
  >(null)
  const drag = useRef<{ x: number; y: number; start: Placement } | null>(null)
  const stage = useRef<HTMLDivElement>(null)

  // Rendered preview frame for the current time and settings.
  useEffect(() => {
    const ctrl = new AbortController()
    const timer = setTimeout(async () => {
      setLoading(true)
      try {
        const blob = await engine.preview(path, time, settings, ctrl.signal)
        setImage((old) => {
          if (old) URL.revokeObjectURL(old)
          return URL.createObjectURL(blob)
        })
        setError(null)
      } catch (e) {
        if (!ctrl.signal.aborted) setError((e as Error).message)
      } finally {
        if (!ctrl.signal.aborted) setLoading(false)
      }
    }, DEBOUNCE_MS)
    return () => {
      clearTimeout(timer)
      ctrl.abort()
    }
  }, [engine, path, time, settings])

  // Untouched frame for the before/after comparison, fetched only while comparing.
  useEffect(() => {
    if (!comparing) return
    const ctrl = new AbortController()
    engine
      .frame(path, time, ctrl.signal)
      .then((blob) =>
        setOriginal((old) => {
          if (old) URL.revokeObjectURL(old)
          return URL.createObjectURL(blob)
        })
      )
      .catch(() => undefined)
    return () => ctrl.abort()
  }, [engine, path, time, comparing])

  function onPointerDown(e: PointerEvent<HTMLDivElement>): void {
    if (!canFrame) return
    e.currentTarget.setPointerCapture(e.pointerId)
    drag.current = { x: e.clientX, y: e.clientY, start: settings.placement }
  }

  function onPointerMove(e: PointerEvent<HTMLDivElement>): void {
    const d = drag.current
    const rect = stage.current?.getBoundingClientRect()
    if (!d || !rect) return
    const clamp = (v: number): number => Math.max(-1, Math.min(1, v))
    onPlacement({
      ...d.start,
      x: clamp(d.start.x + (e.clientX - d.x) / rect.width),
      y: clamp(d.start.y + (e.clientY - d.y) / rect.height)
    })
  }

  function onWheel(e: WheelEvent<HTMLDivElement>): void {
    if (!canFrame) return
    const scale = Math.max(
      0.4,
      Math.min(1.6, settings.placement.scale * Math.exp(-e.deltaY * 0.001))
    )
    onPlacement({ ...settings.placement, scale })
  }

  async function renderClip(): Promise<void> {
    setClip({ state: 'rendering', progress: 0, relighting: false })
    try {
      const duration = Math.max(0.5, Math.min(CLIP_SECONDS, info.duration - time))
      const job = await engine.submitJob({ path, settings, start: time, duration })
      const done = await waitForJob(engine, job.id, (j) =>
        setClip({
          state: 'rendering',
          progress: j.total ? j.frame / j.total : 0,
          relighting: j.stage === 'relight'
        })
      )
      if (done.status !== 'done') throw new Error(done.error ?? `preview ${done.status}`)
      setClip({ state: 'ready', url: engine.mediaUrl('/media', { path: done.out }) })
    } catch (e) {
      setClip(null)
      setError((e as Error).message)
    }
  }

  const shown = comparing && original ? original : image

  return (
    <div className="preview">
      <div
        ref={stage}
        className={`stage ${canFrame ? 'draggable' : ''}`}
        style={{ aspectRatio: `${info.width} / ${info.height}` }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={() => (drag.current = null)}
        onWheel={onWheel}
      >
        {shown ? (
          <img src={shown} alt="Preview" draggable={false} />
        ) : (
          <div className="stage-empty" />
        )}
        {loading && <div className="stage-spinner spinner small" />}
        {comparing && <span className="stage-label">Original</span>}
        {clip?.state === 'ready' && (
          <div className="clip-overlay" onPointerDown={(e) => e.stopPropagation()}>
            <video src={clip.url} autoPlay loop controls />
            <button className="ghost small" onClick={() => setClip(null)}>
              Back to editing
            </button>
          </div>
        )}
      </div>

      <div className="timeline">
        <span className="time">{formatTime(time)}</span>
        <input
          type="range"
          min={0}
          max={info.duration}
          step={1 / info.fps}
          value={time}
          onChange={(e) => setTime(Number(e.target.value))}
        />
        <span className="time muted">{formatTime(info.duration)}</span>
      </div>

      <div className="preview-actions">
        <button
          className="ghost"
          onPointerDown={() => setComparing(true)}
          onPointerUp={() => setComparing(false)}
          onPointerLeave={() => setComparing(false)}
        >
          Hold to compare
        </button>
        <button className="ghost" onClick={renderClip} disabled={clip?.state === 'rendering'}>
          {clip?.state === 'rendering'
            ? `${clip.relighting ? 'Relighting' : 'Rendering preview'}… ${Math.round(clip.progress * 100)}%`
            : `Play ${CLIP_SECONDS}s preview`}
        </button>
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  )
}
