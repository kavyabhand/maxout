/** Placeholder block shown while an artifact-backed figure is loading. */
export function Skeleton({ className = "h-4 w-full" }: { className?: string }) {
  return <div className={`animate-pulse rounded-lg bg-(--color-surface-3) ${className}`} />
}
