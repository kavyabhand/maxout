import clsx from "clsx"
import { useCountUp } from "../../hooks/useCountUp"

/**
 * A measured figure that counts up on arrival. The animation is
 * presentation only; the value it settles on is exactly what the API
 * returned, and it exists because every number on these screens lands
 * at once the moment its fetch resolves, which reads as a static report
 * rather than a live instrument.
 */
export function CountUpValue({
  target,
  decimals = 2,
  suffix = "",
  className,
}: {
  target: number | null
  decimals?: number
  suffix?: string
  className?: string
}) {
  const v = useCountUp(target)
  if (target === null) return <span className={className}>, </span>
  return (
    <span className={clsx("tabular-nums", className)}>
      {v.toFixed(decimals)}
      {suffix}
    </span>
  )
}
