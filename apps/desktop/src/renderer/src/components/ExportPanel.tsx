import { useState, type ReactNode } from 'react'
import { waitForJob, type Engine, type Job, type Settings, type VideoInfo } from '../api'
import { formatTime } from '../format'

export function ExportPanel({
  engine,
  path,
  info,
  settings
}: {
  engine: Engine
  path: string
  info: VideoInfo
  settings: Settings
}): ReactNode {
  const [job, setJob] = useState<Job | null>(null)
  const [error, setError] = useState<string | null>(null)
  const running = job?.status === 'queued' || job?.status === 'running'

  async function start(): Promise<void> {
    const out = await window.vibe.saveVideo(path)
    if (!out) return
    setError(null)
    try {
      const submitted = await engine.submitJob({ path, out, settings })
      setJob(submitted)
      const done = await waitForJob(engine, submitted.id, setJob, 500)
      if (done.status === 'failed') setError(done.error)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const progress = job && job.total ? job.frame / job.total : 0

  return (
    <div className="export">
      {running ? (
        <>
          <div className="progress">
            <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
          </div>
          <div className="export-row">
            <span className="muted">
              {job.status === 'queued'
                ? 'Waiting…'
                : job.stage === 'relight'
                  ? `Relighting keyframes ${job.frame}/${job.total}…`
                  : `${Math.round(progress * 100)}% · ${job.eta != null ? `${formatTime(job.eta)} left` : 'estimating…'}`}
            </span>
            <button className="ghost small" onClick={() => engine.cancelJob(job.id)}>
              Cancel
            </button>
          </div>
        </>
      ) : (
        <button className="primary wide" onClick={start}>
          Export video
        </button>
      )}
      {job?.status === 'done' && !running && (
        <div className="export-row">
          <span className="ok">Exported in {formatTime(job.elapsed)}</span>
          <button className="ghost small" onClick={() => window.vibe.reveal(job.out)}>
            Show file
          </button>
        </div>
      )}
      {job?.status === 'cancelled' && <p className="muted">Export cancelled.</p>}
      {error && <p className="error">{error}</p>}
      <p className="hint">
        {info.width}×{info.height}, same frame rate{info.has_audio ? ' and audio' : ''} as the
        original.
      </p>
    </div>
  )
}
