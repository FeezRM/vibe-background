import type { ReactNode } from 'react'

export function Section({ title, children }: { title: string; children: ReactNode }): ReactNode {
  return (
    <section className="section">
      <h3>{title}</h3>
      {children}
    </section>
  )
}

export function Slider({
  label,
  value,
  min = 0,
  max = 1,
  step = 0.01,
  format = (v) => `${Math.round(v * 100)}`,
  onChange
}: {
  label: string
  value: number
  min?: number
  max?: number
  step?: number
  format?: (v: number) => string
  onChange: (v: number) => void
}): ReactNode {
  return (
    <label className="slider">
      <span className="slider-label">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="slider-value">{format(value)}</span>
    </label>
  )
}

export function Segmented<T extends string>({
  options,
  value,
  onChange
}: {
  options: { value: T; label: string; hint?: string }[]
  value: T
  onChange: (v: T) => void
}): ReactNode {
  return (
    <div className="segmented" role="radiogroup">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={o.value === value}
          className={o.value === value ? 'active' : ''}
          title={o.hint}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Toggle({
  label,
  checked,
  onChange
}: {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
}): ReactNode {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  )
}
