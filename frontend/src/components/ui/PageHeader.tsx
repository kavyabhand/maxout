import type { ReactNode } from "react"

/**
 * Screen header: an eyebrow naming the pillar, a title, and one line.
 * The cap is deliberate; every screen previously opened with a three-
 * or four-line paragraph, and that repeated block of prose was most of
 * what made the app feel content-heavy before the reader had seen a
 * single number.
 */
export function PageHeader({
  pillar,
  color,
  title,
  lede,
  action,
}: {
  pillar: string
  color: string
  title: string
  lede: string
  action?: ReactNode
}) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-x-6 gap-y-2">
      <div className="max-w-2xl">
        <span className="text-[10.5px] font-semibold uppercase tracking-[0.1em]" style={{ color }}>
          {pillar}
        </span>
        <h1 className="mt-1 text-[24px] font-bold leading-tight tracking-tight text-(--color-text-primary)">
          {title}
        </h1>
        <p className="mt-1 text-[14px] leading-relaxed text-(--color-text-secondary)">{lede}</p>
      </div>
      {action}
    </header>
  )
}
