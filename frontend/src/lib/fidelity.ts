/**
 * The distinguisher AUC is the one score in this app where lower is
 * better and 0.5, not 1.0, is perfect. Left as a bare number it reliably
 * reads as "85% good" when it means the opposite, so the verdict is
 * derived in one place and shown wherever the figure is.
 */
export interface FidelityVerdict {
  label: string
  /** Tailwind text-colour class for the verdict word. */
  tone: string
}

export function fidelityVerdict(auc: number): FidelityVerdict {
  const gap = Math.abs(auc - 0.5)
  if (gap < 0.1) return { label: "Near-indistinguishable", tone: "text-(--color-defense)" }
  if (gap < 0.3) return { label: "Close, with tells", tone: "text-(--color-tier-review)" }
  return { label: "Distinguishable", tone: "text-(--color-attack)" }
}
