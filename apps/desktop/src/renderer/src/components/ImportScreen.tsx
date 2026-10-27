import { useState, type DragEvent, type ReactNode } from 'react'
import type { Engine, VideoInfo } from '../api'

export function ImportScreen({
  engine,
  onOpen
}: {
  engine: Engine
  onOpen: (path: string, info: VideoInfo) => void
}): ReactNode {
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function open(path: string | null): Promise<void> {
    if (!path) return
    setBusy(true)
    setError(null)
    try {
      onOpen(path, await engine.probe(path))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  function onDrop(e: DragEvent): void {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) open(window.vibe.pathForFile(file))
  }

  return (
    <div className="import-screen drag-region">
      <div
        className={`dropzone no-drag ${dragging ? 'dragging' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <h1>Vibe Background</h1>
        <p className="muted">
          Drop a video of yourself. We’ll cut you out and put you somewhere nicer.
        </p>
        <button
          className="primary"
          disabled={busy}
          onClick={async () => open(await window.vibe.openVideo())}
        >
          {busy ? 'Opening…' : 'Choose video'}
        </button>
        {error && <p className="error">{error}</p>}
        <p className="fine-print">Everything runs on this computer. Nothing is uploaded.</p>
      </div>
    </div>
  )
}
