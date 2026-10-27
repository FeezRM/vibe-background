import { useEffect, useState, type ReactNode } from 'react'
import { Engine, type VideoInfo } from './api'
import { Editor } from './components/Editor'
import { ImportScreen } from './components/ImportScreen'

type Connection =
  { state: 'connecting' } | { state: 'ready'; engine: Engine } | { state: 'error'; message: string }

function App(): ReactNode {
  const [conn, setConn] = useState<Connection>({ state: 'connecting' })
  const [video, setVideo] = useState<{ path: string; info: VideoInfo } | null>(null)

  useEffect(() => {
    window.vibe
      .connectEngine()
      .then(({ url, token }) => setConn({ state: 'ready', engine: new Engine(url, token) }))
      .catch((e: Error) => setConn({ state: 'error', message: e.message }))
  }, [])

  if (conn.state === 'connecting') {
    return (
      <div className="center-screen">
        <div className="spinner" />
        <p className="muted">Starting the engine…</p>
      </div>
    )
  }
  if (conn.state === 'error') {
    return (
      <div className="center-screen">
        <h2>The engine didn’t start</h2>
        <pre className="error-log">{conn.message}</pre>
      </div>
    )
  }
  if (!video) {
    return <ImportScreen engine={conn.engine} onOpen={(path, info) => setVideo({ path, info })} />
  }
  return (
    <Editor
      key={video.path}
      engine={conn.engine}
      path={video.path}
      info={video.info}
      onClose={() => setVideo(null)}
    />
  )
}

export default App
