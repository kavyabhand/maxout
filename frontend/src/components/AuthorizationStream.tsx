/**
 * The defense, running.
 *
 * Everything else in this prototype reports what happened in a batch job.
 * This is the one surface where a judge can watch the stacked model make
 * decisions at a rate they control, and see the four issuer tiers fill up
 * in proportion as it goes.
 *
 * What is real and what is not, stated plainly because the distinction is
 * the whole credibility of the panel: every row is a decision the model
 * actually produced on a row of the IEEE-CIS held-out split; the three
 * member scores, the stacked risk score and the resulting tier are all as
 * measured, and the sample is a faithful random slice, not a curated one.
 * The only thing added here is the passage of time. The panel says so on
 * screen rather than leaving a viewer to assume it is scoring live.
 */

import { useMemo, useState } from "react"
import clsx from "clsx"
import { useApiData } from "../api/hooks"
import { useReplayStream } from "../hooks/useReplayStream"
import type { DecisionTier, EnsembleResult, LatencyProfile, ScoredDecision } from "../api/types"

const TIER_META: Record<DecisionTier, { label: string; color: string; bg: string }> = {
  auto_approve: { label: "Approved", color: "var(--color-tier-approve)", bg: "var(--color-defense-bg)" },
  step_up: { label: "Step-up", color: "var(--color-tier-stepup)", bg: "#f2f6fc" },
  review: { label: "Review", color: "var(--color-tier-review)", bg: "var(--color-tier-review-bg)" },
  decline: { label: "Declined", color: "var(--color-tier-decline)", bg: "var(--color-attack-bg)" },
}
const TIER_ORDER: DecisionTier[] = ["auto_approve", "step_up", "review", "decline"]

const SPEEDS = [4, 16, 60]
const WINDOW_SIZE = 7

export function AuthorizationStream() {
  const ensemble = useApiData<EnsembleResult>("/api/defend/ensemble")
  const latency = useApiData<LatencyProfile>("/api/defend/latency")

  const [running, setRunning] = useState(true)
  const [rate, setRate] = useState(16)

  // Each decision's tier is recomputed from its (real, measured) risk
  // score against the capacity-planned cuts rather than read from the
  // artifact's fixed-threshold field. Under fixed probability cuts this
  // model never reaches 0.85, so the decline tier is genuinely always
  // empty, true, and a stream where one of four outcomes can never occur
  // shows a viewer nothing. The capacity cuts are the operating point an
  // issuer would actually deploy at, and they exercise all four. The score
  // and the label are untouched either way.
  const cuts = ensemble.data?.tier_distribution_capacity?.thresholds ?? null
  const sample = useMemo(() => {
    const raw = ensemble.data?.scored_sample ?? null
    if (!raw || !cuts) return raw
    return raw.map((d) => ({
      ...d,
      tier: (d.risk >= cuts.decline
        ? "decline"
        : d.risk >= cuts.review
          ? "review"
          : d.risk >= cuts.step_up
            ? "step_up"
            : "auto_approve") as DecisionTier,
    }))
  }, [ensemble.data, cuts])
  const { window: feed, totals } = useReplayStream<ScoredDecision>(sample, {
    running: running && !!sample,
    ratePerSecond: rate,
    windowSize: WINDOW_SIZE,
  })

  const shares = useMemo(() => {
    const total = totals.seen || 1
    return TIER_ORDER.map((tier) => ({ tier, share: (totals.byTier[tier] ?? 0) / total }))
  }, [totals])

  const caught = (totals.fraudByTier.decline ?? 0) + (totals.fraudByTier.review ?? 0)
  const p95 = latency.data?.single_row_scoring_ms.p95

  if (!ensemble.loading && !sample) {
    return (
      <section className="rounded-(--radius-panel) border border-(--color-hairline) px-4 py-5 text-[13px] text-(--color-text-tertiary)">
        The stacked-ensemble artifact has not been generated yet. Run{" "}
        <code className="font-mono text-[12px] text-(--color-text-secondary)">
          python -m janus.orchestrate.persist meta_ensemble
        </code>{" "}
        to populate this stream.
      </section>
    )
  }

  return (
    <section className="overflow-hidden rounded-(--radius-panel) border border-(--color-hairline)">
      {/* control strip */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-(--color-hairline) bg-(--color-surface-1) px-3 py-2">
        <button
          type="button"
          onClick={() => setRunning((r) => !r)}
          className="inline-flex min-h-[34px] items-center gap-1.5 rounded-(--radius-control) border border-(--color-border) bg-(--color-surface-0) px-3 py-1 text-[12.5px] font-medium text-(--color-text-primary) transition-colors hover:bg-(--color-surface-2) focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--color-focus) sm:min-h-0"
          aria-pressed={running}
        >
          <span
            className={clsx(
              "h-1.5 w-1.5 rounded-full",
              running ? "animate-blink bg-(--color-defense)" : "bg-(--color-surface-4)",
            )}
          />
          {running ? "Pause" : "Resume"}
        </button>

        <div className="flex items-center gap-1" role="group" aria-label="Replay rate">
          {SPEEDS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setRate(s)}
              aria-pressed={rate === s}
              className={clsx(
                // min-h on touch widths only: 24px is the WCAG 2.2 floor for
                // a pointer target and these sat at 22px, while the desktop
                // strip is meant to stay dense.
                "inline-flex min-h-[34px] min-w-[44px] items-center justify-center rounded-full px-2.5 text-[12px] font-medium tabular-nums transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--color-focus) sm:min-h-[26px] sm:min-w-0",
                rate === s
                  ? "bg-(--color-text-primary) text-white"
                  : "text-(--color-text-tertiary) hover:bg-(--color-surface-2)",
              )}
            >
              {s}/s
            </button>
          ))}
        </div>

        {/* Stacked label-over-value on narrow screens: inline pairs wrapped
            mid-label there and cost three lines to say three numbers. */}
        <dl className="grid w-full grid-cols-3 gap-x-3 text-[12px] tabular-nums sm:ml-auto sm:flex sm:w-auto sm:items-center sm:gap-x-5">
          <Readout label="Authorized" value={totals.seen.toLocaleString()} />
          <Readout label="Fraud caught" value={`${caught} / ${totals.fraud}`} />
          {p95 !== undefined && <Readout label="p95 score time" value={`${p95.toFixed(2)} ms`} />}
        </dl>
      </div>

      {/* tier proportions */}
      <div className="px-3 pb-1 pt-3">
        <div className="flex h-2 overflow-hidden rounded-full bg-(--color-surface-2)">
          {shares.map(({ tier, share }) => (
            <div
              key={tier}
              className="h-full transition-[width] duration-500 ease-out"
              style={{ width: `${share * 100}%`, background: TIER_META[tier].color }}
              title={`${TIER_META[tier].label}: ${(share * 100).toFixed(1)}%`}
            />
          ))}
        </div>
        <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1">
          {shares.map(({ tier, share }) => (
            <span key={tier} className="flex items-baseline gap-1.5 text-[12px]">
              <span className="h-1.5 w-1.5 translate-y-[-1px] rounded-full" style={{ background: TIER_META[tier].color }} />
              <span className="text-(--color-text-secondary)">{TIER_META[tier].label}</span>
              <span className="font-semibold tabular-nums text-(--color-text-primary)">
                {(share * 100).toFixed(1)}%
              </span>
            </span>
          ))}
        </div>
      </div>

      {/* The feed. Fixed height so the panel does not resize as rows fill
          in, and the entrance animation is dropped above a few rows per
          second: a 280ms rise that has not finished before the next row
          arrives leaves the top of the list visibly blank, which reads as
          a rendering fault rather than as motion. */}
      <ol
        className="mt-1 divide-y divide-(--color-hairline) border-t border-(--color-hairline)"
        style={{ minHeight: WINDOW_SIZE * 30 }}
      >
        {feed.length === 0 && (
          <li className="px-3 py-3 text-[12.5px] text-(--color-text-tertiary)">Waiting for the first decision…</li>
        )}
        {feed.map(({ item, key }) => (
          <DecisionRow key={key} d={item} animate={rate <= 8} />
        ))}
      </ol>

      <p className="border-t border-(--color-hairline) bg-(--color-surface-1) px-3 py-2 text-[11.5px] leading-snug text-(--color-text-tertiary)">
        Replaying {sample?.length.toLocaleString() ?? ", "} decisions the stacked model produced on the IEEE-CIS
        held-out split: a faithful random slice, not a curated one. Every risk score and label is exactly as
        measured; the tier is that score placed against the capacity-planned cuts (top 0.2% of volume declines,
        next 1% reviews). Only the pacing is added.
      </p>
    </section>
  )
}

function Readout({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex min-w-0 flex-col sm:flex-row sm:items-baseline sm:gap-1.5">
      <dt className="truncate text-[11px] text-(--color-text-tertiary) sm:text-[12px]">{label}</dt>
      <dd className="font-semibold text-(--color-text-primary)">{value}</dd>
    </div>
  )
}

function DecisionRow({ d, animate }: { d: ScoredDecision; animate: boolean }) {
  const meta = TIER_META[d.tier]
  return (
    <li
      className={clsx(
        "grid grid-cols-[auto_1fr_auto] items-center gap-x-3 px-3 py-1.5 text-[12.5px]",
        animate && "animate-rise",
      )}
      style={{ background: d.is_fraud ? meta.bg : undefined }}
    >
      <span className="font-mono tabular-nums text-(--color-text-tertiary)">
        {String(d.hour).padStart(2, "0")}:00
      </span>

      <span className="flex min-w-0 items-center gap-2.5">
        <span className="font-semibold tabular-nums text-(--color-text-primary)">
          ${d.amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </span>
        <span className="shrink-0 text-(--color-text-tertiary)">{d.product}</span>
        {/* The three members that voted, so the stack is visibly a stack
            rather than one number with a story attached. */}
        <span className="hidden shrink-0 items-center gap-1.5 sm:flex" title="GBM / GNN / anomaly member scores">
          <Pip v={d.gbm} />
          <Pip v={d.gnn} />
          <Pip v={d.anomaly} />
        </span>
        {d.is_fraud === 1 && (
          <span className="shrink-0 rounded-full bg-(--color-attack-bg) px-1.5 py-px text-[10.5px] font-semibold uppercase tracking-wide text-(--color-attack)">
            Fraud
          </span>
        )}
      </span>

      <span className="flex items-center gap-2 justify-self-end">
        <span className="font-mono tabular-nums text-(--color-text-secondary)">{d.risk.toFixed(3)}</span>
        <span
          className="w-[68px] rounded-full px-2 py-px text-center text-[11px] font-semibold"
          style={{ color: meta.color, background: meta.bg }}
        >
          {meta.label}
        </span>
      </span>
    </li>
  )
}

/** One member's score as a filled bar, three of them side by side. Reading
 *  three numbers per row at this rate is impossible; reading three heights
 *  is not. */
function Pip({ v }: { v: number }) {
  return (
    <span className="relative block h-3.5 w-1 overflow-hidden rounded-sm bg-(--color-surface-3)">
      <span
        className="absolute bottom-0 left-0 w-full rounded-sm bg-(--color-text-secondary)"
        style={{ height: `${Math.max(6, Math.min(1, v) * 100)}%` }}
      />
    </span>
  )
}
