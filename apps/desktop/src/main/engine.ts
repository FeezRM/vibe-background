import { ChildProcess, spawn } from 'child_process'
import { join, resolve } from 'path'
import { app } from 'electron'
import { is } from '@electron-toolkit/utils'

export interface EngineInfo {
  url: string
  token: string
}

const READY_PREFIX = 'VIBEBG_READY '

/** Starts the Python engine and resolves once it announces its port and token. */
export class Engine {
  private proc: ChildProcess | null = null
  private info: Promise<EngineInfo> | null = null
  private log: string[] = []

  start(): Promise<EngineInfo> {
    if (this.info) return this.info
    const userPresets = join(app.getPath('userData'), 'presets')
    const { command, args, cwd } = this.command(userPresets)

    this.info = new Promise((resolvePromise, reject) => {
      const proc = spawn(command, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'] })
      this.proc = proc
      let buffered = ''
      let ready = false

      proc.stdout!.on('data', (chunk: Buffer) => {
        buffered += chunk.toString()
        let nl: number
        while ((nl = buffered.indexOf('\n')) >= 0) {
          const line = buffered.slice(0, nl).trim()
          buffered = buffered.slice(nl + 1)
          if (!ready && line.startsWith(READY_PREFIX)) {
            const { port, token } = JSON.parse(line.slice(READY_PREFIX.length))
            ready = true
            resolvePromise({ url: `http://127.0.0.1:${port}`, token })
          } else if (line) {
            this.remember(line)
          }
        }
      })
      proc.stderr!.on('data', (chunk: Buffer) => this.remember(chunk.toString()))
      proc.on('error', (err) => reject(new Error(`could not start the engine: ${err.message}`)))
      proc.on('exit', (code) => {
        this.proc = null
        if (!ready) {
          reject(new Error(`the engine exited (code ${code}):\n${this.log.slice(-20).join('\n')}`))
        }
        this.info = null
      })
    })
    return this.info
  }

  stop(): void {
    this.proc?.kill()
    this.proc = null
  }

  private remember(text: string): void {
    this.log.push(text)
    if (this.log.length > 200) this.log.splice(0, this.log.length - 200)
    if (is.dev) process.stderr.write(`[engine] ${text}\n`)
  }

  private command(userPresets: string): { command: string; args: string[]; cwd: string } {
    if (is.dev) {
      // apps/desktop -> repo root, where engine/ and presets/ live.
      const root = resolve(app.getAppPath(), '..', '..')
      return {
        command: 'uv',
        args: [
          'run',
          '--project',
          join(root, 'engine'),
          'vibebg-server',
          '--presets',
          join(root, 'presets', 'backgrounds'),
          '--user-presets',
          userPresets
        ],
        cwd: root
      }
    }
    const runtime = join(app.getPath('userData'), 'engine')
    const bin =
      process.platform === 'win32'
        ? join(runtime, '.venv', 'Scripts', 'vibebg-server.exe')
        : join(runtime, '.venv', 'bin', 'vibebg-server')
    return {
      command: bin,
      args: [
        '--presets',
        join(process.resourcesPath, 'presets', 'backgrounds'),
        '--user-presets',
        userPresets
      ],
      cwd: runtime
    }
  }
}
