import { useState, type ReactNode } from 'react'
import type { Engine, PresetInfo, Settings } from '../api'
import { Section, Slider } from './controls'

export function BackgroundPanel({
  engine,
  presets,
  error,
  settings,
  reloadPresets,
  onChange
}: {
  engine: Engine
  presets: PresetInfo[]
  error: string | null
  settings: Settings
  reloadPresets: () => Promise<PresetInfo[]>
  onChange: (patch: (s: Settings) => Settings) => void
}): ReactNode {
  const [importing, setImporting] = useState(false)
  const [importError, setImportError] = useState<string | null>(null)
  const bg = settings.background
  const selected = bg.mode === 'preset' ? presets.find((p) => p.id === bg.preset) : undefined

  function choosePreset(p: PresetInfo): void {
    onChange((s) => ({
      ...s,
      // The blur slider means different things per background, so each choice starts fresh.
      background: { ...s.background, mode: 'preset', preset: p.id, blur: 0 },
      placement: { ...p.subject },
      lighting: { ...s.lighting, shadow: p.light.shadow, warmth: p.light.warmth }
    }))
  }

  async function upload(): Promise<void> {
    const path = await window.vibe.openBackground()
    if (!path) return
    setImporting(true)
    setImportError(null)
    try {
      const name = (path.split(/[\\/]/).pop() ?? 'Background').replace(/\.[^.]+$/, '')
      const { id } = await engine.addPreset(path, name)
      const list = await reloadPresets()
      const added = list.find((p) => p.id === id)
      if (added) choosePreset(added)
    } catch (e) {
      setImportError((e as Error).message)
    } finally {
      setImporting(false)
    }
  }

  async function remove(p: PresetInfo): Promise<void> {
    if (!confirm(`Remove “${p.name}” from your backgrounds?`)) return
    await engine.deletePreset(p.id)
    if (bg.preset === p.id)
      onChange((s) => ({ ...s, background: { ...s.background, mode: 'blur_original' } }))
    await reloadPresets()
  }

  return (
    <Section title="Background">
      <div className="preset-grid">
        <button
          className={`preset-tile blur-tile ${bg.mode === 'blur_original' ? 'selected' : ''}`}
          onClick={() =>
            onChange((s) => ({
              ...s,
              background: { ...s.background, mode: 'blur_original', blur: 0.5 },
              placement: { scale: 1, x: 0, y: 0 }
            }))
          }
        >
          <span>Blur my room</span>
        </button>
        {presets.map((p) => (
          <div
            key={p.id}
            className={`preset-tile ${bg.preset === p.id && bg.mode === 'preset' ? 'selected' : ''}`}
          >
            <button className="preset-thumb" onClick={() => choosePreset(p)} title={p.name}>
              <img src={engine.mediaUrl(`/presets/${p.id}/thumb`)} alt="" loading="lazy" />
              {p.type === 'video' && <span className="badge">Loop</span>}
            </button>
            <span className="preset-name">{p.name}</span>
            {!p.builtin && (
              <button className="remove" title="Remove" onClick={() => remove(p)}>
                ×
              </button>
            )}
          </div>
        ))}
        <button className="preset-tile upload-tile" onClick={upload} disabled={importing}>
          <span>{importing ? 'Adding…' : '+ Your own'}</span>
        </button>
      </div>
      {(error || importError) && <p className="error">{error || importError}</p>}

      {bg.mode === 'blur_original' && (
        <Slider
          label="Blur"
          value={bg.blur}
          onChange={(blur) => onChange((s) => ({ ...s, background: { ...s.background, blur } }))}
        />
      )}
      {selected?.type === 'image' && (
        <Slider
          label="Depth of field"
          value={bg.dof}
          onChange={(dof) => onChange((s) => ({ ...s, background: { ...s.background, dof } }))}
        />
      )}
      {selected?.type === 'video' && (
        <Slider
          label="Blur"
          value={bg.blur}
          onChange={(blur) => onChange((s) => ({ ...s, background: { ...s.background, blur } }))}
        />
      )}
      {selected && selected.license !== 'user-provided' && (
        <p className="hint credit">
          {selected.url ? (
            <a href={selected.url} target="_blank" rel="noreferrer">
              {selected.source}
            </a>
          ) : (
            selected.source
          )}{' '}
          · {selected.license}
        </p>
      )}
    </Section>
  )
}
