import clsx from "clsx"
import type { AttackStatus } from "../../api/types"

const statusStyle: Record<AttackStatus, string> = {
  simulated: "bg-(--color-defense-bg) text-(--color-defense-strong) border-(--color-defense-dim)",
  modeled: "bg-(--color-tier-review-bg) text-(--color-tier-review) border-[#f2d9a8]",
  taxonomy_only: "bg-(--color-surface-2) text-(--color-text-tertiary) border-(--color-border)",
}

const statusLabel: Record<AttackStatus, string> = {
  simulated: "Simulated",
  modeled: "Modeled",
  taxonomy_only: "Mapped only",
}

export function StatusBadge({ status }: { status: AttackStatus }) {
  return (
    <span
      className={clsx(
        "inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold",
        statusStyle[status],
      )}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {statusLabel[status]}
    </span>
  )
}
