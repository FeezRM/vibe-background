import { useCallback, useEffect, useState, type ReactNode } from 'react'
import {
  DEFAULT_SETTINGS,
  type Engine,
  type PresetInfo,
  type Settings,
  type VideoInfo
} from '../api'
import { BackgroundPanel } from './BackgroundPanel'
import { Section, Segmented, Slider, Toggle } from './controls'
import { ExportPanel } from './ExportPanel'
import { Preview } from './Preview'

export function Editor({
  engine,
  path,
  info,
  onClose
}: {
  engine: Engine
  path: string
  info: VideoInfo
  onClose: () => void
}): ReactNode {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS)
  const [presets, setPresets] = useState<PresetInfo[]>([])
  const [presetError, setPresetError] = useState<string | null>(null)

  const reloadPresets = useCallback(async (): Promise<PresetInfo[]> => {
    try {
      const list = await engine.presets()
      setPresets(list)
      setPresetError(null)
      return list
    } catch (e) {
      setPresetError((e as Error).message)
      return []
    }
  }, [engine])

  useEffect(() => {
    engine
      .presets()
      .then(setPresets)
      .catch((e: Error) => setPresetError(e.message))
  }, [engine])

  const update = (patch: (s: Settings) => Settings): void => setSettings((s) => patch(s))
  const light = settings.lighting
  const canFrame = settings.background.mode !== 'blur_original'
  const fileName = path.split(/[\\/]/).pop()

  return (
    <div className="editor">
      <header className="topbar drag-region">
        <div className="title">
          <span className="file-name">{fileName}</span>
          <span className="muted">
            {info.width}×{info.height} · {info.fps.toFixed(2).replace(/\.00$/, '')} fps ·{' '}
            {info.duration.toFixed(1)}s{info.has_audio ? '' : ' · no audio'}
          </span>
        </div>
        <button className="ghost no-drag" onClick={onClose}>
          Open another video
        </button>
      </header>

      <main className="workspace">
        <Preview
          engine={engine}
          path={path}
          info={info}
          settings={settings}
          canFrame={canFrame}
          onPlacement={(placement) => update((s) => ({ ...s, placement }))}
        />

        <aside className="sidebar">
          <BackgroundPanel
            engine={engine}
            presets={presets}
            error={presetError}
            settings={settings}
            reloadPresets={reloadPresets}
            onChange={update}
          />

          {canFrame && (
            <Section title="Framing">
              <Slider
                label="Size"
                value={settings.placement.scale}
                min={0.4}
                max={1.6}
                format={(v) => `${Math.round(v * 100)}%`}
                onChange={(scale) =>
                  update((s) => ({ ...s, placement: { ...s.placement, scale } }))
                }
              />
              <p className="hint">Drag the preview to move yourself.</p>
              <button
                className="ghost small"
                onClick={() => update((s) => ({ ...s, placement: { scale: 1, x: 0, y: 0 } }))}
              >
                Reset framing
              </button>
            </Section>
          )}

          <Section title="Lighting">
            <Segmented
              value={light.mode}
              onChange={(mode) => update((s) => ({ ...s, lighting: { ...s.lighting, mode } }))}
              options={[
                { value: 'off', label: 'Off' },
                { value: 'enhance', label: 'Enhance', hint: 'Fix exposure and color on you' },
                { value: 'match', label: 'Match', hint: 'Light you like the background' },
                {
                  value: 'relight',
                  label: 'AI Relight',
                  hint: 'Experimental: re-light you with AI'
                }
              ]}
            />
            {light.mode === 'relight' && (
              <p className="hint">
                Experimental. An AI model re-lights you to fit the scene, which takes a few seconds
                per preview. The first use downloads about 4 GB of models.
              </p>
            )}
            {light.mode !== 'off' && (
              <Slider
                label="Strength"
                value={light.strength}
                onChange={(strength) =>
                  update((s) => ({ ...s, lighting: { ...s.lighting, strength } }))
                }
              />
            )}
            {(light.mode === 'match' || light.mode === 'relight') && (
              <>
                <Slider
                  label="Light wrap"
                  value={light.wrap}
                  onChange={(wrap) => update((s) => ({ ...s, lighting: { ...s.lighting, wrap } }))}
                />
                <Slider
                  label="Shadow"
                  value={light.shadow}
                  onChange={(shadow) =>
                    update((s) => ({ ...s, lighting: { ...s.lighting, shadow } }))
                  }
                />
              </>
            )}
            <Slider
              label="Warmth"
              value={light.warmth}
              min={-1}
              max={1}
              format={(v) => (v > 0 ? `+${Math.round(v * 100)}` : `${Math.round(v * 100)}`)}
              onChange={(warmth) => update((s) => ({ ...s, lighting: { ...s.lighting, warmth } }))}
            />
            <Toggle
              label="Reduce noise"
              checked={light.denoise}
              onChange={(denoise) =>
                update((s) => ({ ...s, lighting: { ...s.lighting, denoise } }))
              }
            />
          </Section>

          <Section title="Quality">
            <Segmented
              value={settings.tier}
              onChange={(tier) => update((s) => ({ ...s, tier }))}
              options={[
                { value: 'fast', label: 'Fast', hint: 'Quickest export' },
                { value: 'balanced', label: 'Balanced', hint: 'Cleaner hair and hands' },
                { value: 'max', label: 'Max', hint: 'Finest hair detail' }
              ]}
            />
            {settings.tier === 'max' && (
              <p className="hint">
                Re-traces hair strand by strand. Exports take about 4× longer than Balanced.
              </p>
            )}
          </Section>

          <ExportPanel engine={engine} path={path} info={info} settings={settings} />
        </aside>
      </main>
    </div>
  )
}
