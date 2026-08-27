import { useEffect, useRef, useState } from "react"

/**
 * Animates a number up from zero when it first arrives.
 *
 * Metrics on these screens land all at once the moment their fetch
 * resolves, which reads as a static report. Counting them up gives the
 * dashboard the sense of being live without faking any data; the final
 * value is always exactly what the API returned.
 *
 * Honours prefers-reduced-motion by snapping straight to the target.
 */
export function useCountUp(target: number | null, durationMs = 900): number {
  const [value, setValue] = useState(0)
  const frameRef = useRef<number | null>(null)

  useEffect(() => {
    if (target === null || !Number.isFinite(target)) {
      setValue(0)
      return
    }

    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches
    if (reduced || durationMs <= 0) {
      setValue(target)
      return
    }

    const start = performance.now()
    const tick = (now: number) => {
      const t = Math.min((now - start) / durationMs, 1)
      // easeOutCubic: fast to begin with, settling gently on the real value.
      const eased = 1 - Math.pow(1 - t, 3)
      setValue(target * eased)
      if (t < 1) frameRef.current = requestAnimationFrame(tick)
    }
    frameRef.current = requestAnimationFrame(tick)

    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current)
    }
  }, [target, durationMs])

  return value
}
