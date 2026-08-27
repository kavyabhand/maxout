/**
 * Drives the authorization stream: walks a fixed list of already-scored
 * decisions forward in time and reports both a rolling window of the most
 * recent ones and the running totals.
 *
 * The decisions themselves are real; each one is an output the stacked
 * model produced on a row of the IEEE-CIS held-out split, so nothing here
 * invents a number. What this hook adds is only the passage of time: a
 * batch score is a static table, and a table cannot show a judge what
 * running traffic through this thing looks like.
 *
 * Two details that matter for it not to feel like a toy:
 *
 * - The interval is driven by `requestAnimationFrame` deltas rather than
 *   `setInterval`, so a background tab that throttles timers resumes
 *   without a burst of catch-up events, and the arrival rate stays honest
 *   at high speeds where setInterval's clamped minimum would silently cap
 *   it.
 * - It loops. The sample is finite and a demo is not, so on reaching the
 *   end it wraps and keeps the cumulative counters running rather than
 *   stopping dead mid-presentation.
 */

import { useEffect, useRef, useState } from "react"

export interface ReplayTotals {
  seen: number
  fraud: number
  byTier: Record<string, number>
  fraudByTier: Record<string, number>
}

const EMPTY_TOTALS: ReplayTotals = { seen: 0, fraud: 0, byTier: {}, fraudByTier: {} }

export function useReplayStream<T extends { tier: string; is_fraud: number }>(
  items: T[] | null,
  {
    running,
    ratePerSecond,
    windowSize = 8,
  }: { running: boolean; ratePerSecond: number; windowSize?: number },
): { window: { item: T; key: number }[]; totals: ReplayTotals } {
  const [visible, setVisible] = useState<{ item: T; key: number }[]>([])
  const [totals, setTotals] = useState<ReplayTotals>(EMPTY_TOTALS)

  const cursor = useRef(0)
  const emitted = useRef(0)
  const carry = useRef(0)

  // Reset when the backing sample arrives or is replaced.
  useEffect(() => {
    cursor.current = 0
    emitted.current = 0
    carry.current = 0
    setVisible([])
    setTotals(EMPTY_TOTALS)
  }, [items])

  useEffect(() => {
    if (!running || !items || items.length === 0) return

    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches

    const advance = (dt: number) => {
      carry.current += dt * ratePerSecond
      const due = Math.floor(carry.current)
      if (due <= 0) return
      carry.current -= due

      // Cap the burst so a long frame gap cannot dump hundreds of rows
      // into a single render.
      const batch = Math.min(due, 40)
      const drawn: T[] = []
      for (let i = 0; i < batch; i += 1) {
        drawn.push(items[cursor.current % items.length])
        cursor.current += 1
      }

      setTotals((prev) => {
        const byTier = { ...prev.byTier }
        const fraudByTier = { ...prev.fraudByTier }
        let fraud = prev.fraud
        for (const d of drawn) {
          byTier[d.tier] = (byTier[d.tier] ?? 0) + 1
          if (d.is_fraud) {
            fraud += 1
            fraudByTier[d.tier] = (fraudByTier[d.tier] ?? 0) + 1
          }
        }
        return { seen: prev.seen + drawn.length, fraud, byTier, fraudByTier }
      })

      setVisible((prev) =>
        [...drawn.map((item) => ({ item, key: emitted.current++ })).reverse(), ...prev].slice(0, windowSize),
      )
    }

    if (reduced) {
      // With reduced motion requested the figures still have to advance --
      // they are information, not decoration, just on a plain one-second
      // cadence instead of a per-frame one.
      const id = window.setInterval(() => advance(1), 1000)
      return () => window.clearInterval(id)
    }

    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      advance(Math.min((now - last) / 1000, 0.5))
      last = now
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [items, running, ratePerSecond, windowSize])

  return { window: visible, totals }
}
