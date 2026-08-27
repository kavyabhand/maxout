/**
 * Live: the landing screen, and the demo itself.
 *
 * A judge arriving cold should meet the working system, not a description
 * of it, so this screen is the two halves of the loop side by side and
 * both of them moving:
 *
 *   Blue team, always running.  The stacked ensemble's decisions replay at
 *   a rate the viewer controls, filling the four issuer tiers in real
 *   proportion. It is alive the moment the page loads, with nothing to
 *   click.
 *   Red team, on demand.        A real prompt injection driven through the
 *   AP2 shopping-agent sandbox, with the Mandate Firewall on or off. This
 *   one waits for a decision, because the interesting part is comparing
 *   the two outcomes.
 *
 * Beneath them, one row of figures, each labelled with the criterion the
 * challenge is judged on and each read live from the artifact written by
 * the pipeline that produced it, so nothing on this screen can drift
 * from what was actually measured.
 */

import type { ReactNode } from "react"
import { Link } from "react-router-dom"
import { useApiData } from "../api/hooks"
import type { ClosedLoopResult, CoverageSummary, EnsembleResult, FidelityScorecard, GbmMetrics } from "../api/types"
import { AuthorizationStream } from "../components/AuthorizationStream"
import { LiveSandbox } from "../components/LiveSandbox"
import { CountUpValue } from "../components/ui/CountUpValue"
import { fidelityVerdict } from "../lib/fidelity"

export function LiveScreen() {
  const coverage = useApiData<CoverageSummary>("/api/identify/coverage")
  const gbm = useApiData<GbmMetrics>("/api/defend/gbm")
  const ensemble = useApiData<EnsembleResult>("/api/defend/ensemble")
  const loop = useApiData<ClosedLoopResult>("/api/orchestrate/closed-loop")
  const fidelity = useApiData<FidelityScorecard[]>("/api/generate/fidelity")

  // The fraud class is the headline: it is the harder of the two to
  // generate (492 real rows to fit on) and the one a defense is actually
  // trained against, so leading with the easier legitimate batch would be
  // choosing the flattering number.
  const headline =
    fidelity.data?.find((f) => f.batch_name === "tabular_synthesis_ulb_fraud") ??
    fidelity.data?.find((f) => f.batch_name === "tabular_synthesis_ulb") ??
    fidelity.data?.[0]
  const rounds = loop.data?.tabular_adversarial?.rounds ?? []
  const lastEvasion = rounds.length ? rounds[rounds.length - 1].evasion_rate : null
  const fidelityAuc = headline?.distinguisher?.auc ?? null
  const verdict = fidelityAuc === null ? null : fidelityVerdict(fidelityAuc)

  // The stacked ensemble is the headline detector once it has been run;
  // the single-family GBM baseline stands in until then rather than
  // leaving the tile blank.
  const detection = ensemble.data?.stacked ?? gbm.data ?? null
  const detectionFoot = ensemble.data
    ? `Five families stacked, on ${ensemble.data.n_test.toLocaleString()} held-out transactions`
    : gbm.data
      ? `${gbm.data.n_rows.toLocaleString()} transactions, ${gbm.data.n_fraud} fraudulent`
      : undefined

  return (
    <div className="mx-auto flex max-w-[1240px] flex-col gap-7 px-4 py-6 sm:px-6">
      <header className="flex flex-wrap items-end justify-between gap-x-6 gap-y-2">
        <div>
          <h1 className="text-[26px] font-bold leading-tight tracking-tight text-(--color-text-primary)">
            Both sides of the loop, running
          </h1>
          <p className="mt-1 text-[14px] text-(--color-text-secondary)">
            A defense scoring live payment traffic, and a GenAI red team attacking a payment agent.
          </p>
        </div>
        <LoopChips />
      </header>

      <Half
        eyebrow="Blue team"
        title="Authorization stream"
        note="Running continuously"
        tone="var(--color-defense)"
      >
        <AuthorizationStream />
      </Half>

      <Half eyebrow="Red team" title="Attack sandbox" note="Pick a technique and run it" tone="var(--color-attack)">
        <LiveSandbox />
      </Half>

      <section>
        <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-[0.1em] text-(--color-text-tertiary)">
          Measured evidence
        </h2>
        <div className="grid grid-cols-1 gap-px overflow-hidden rounded-(--radius-panel) border border-(--color-hairline) bg-(--color-hairline) sm:grid-cols-2 xl:grid-cols-4">
          <Figure
            criterion="Diversity"
            to="/atlas"
            value={coverage.data?.total_attacks ?? null}
            decimals={0}
            caption="GenAI-era fraud vectors mapped"
            foot={
              coverage.data
                ? `${coverage.data.by_status.simulated} generated end to end, across all four categories`
                : undefined
            }
          />
          {/* Stated with its verdict attached, because this is the one
              figure here where lower is better and 0.500 rather than
              1.000 is perfect, on its own it reads as very nearly the
              opposite of what it means. */}
          <Figure
            criterion="Fidelity"
            to="/studio"
            value={fidelityAuc}
            decimals={3}
            caption="Real-vs-synthetic detector AUC"
            foot={
              verdict ? (
                <>
                  <span className={`font-semibold ${verdict.tone}`}>{verdict.label}</span>, 0.500 would mean a
                  classifier cannot tell them apart
                </>
              ) : undefined
            }
          />
          <Figure
            criterion="Detection efficacy"
            to="/defense"
            value={detection?.pr_auc ?? null}
            decimals={3}
            caption="PR-AUC on real card traffic"
            foot={detectionFoot}
          />
          <Figure
            criterion="Closed loop"
            to="/arena"
            value={lastEvasion !== null ? lastEvasion * 100 : null}
            decimals={0}
            suffix="%"
            caption="Evasion rate after hardening"
            foot={
              rounds.length ? `down from ${(rounds[0].evasion_rate * 100).toFixed(0)}% before the loop ran` : undefined
            }
          />
        </div>
      </section>
    </div>
  )
}

/** A labelled half of the loop. The eyebrow carries the colour so the two
 *  sections read as opposed teams without either one needing a border,
 *  a card or a paragraph of setup. */
function Half({
  eyebrow,
  title,
  note,
  tone,
  children,
}: {
  eyebrow: string
  title: string
  note: string
  tone: string
  children: ReactNode
}) {
  return (
    <section>
      <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span
          className="text-[11px] font-semibold uppercase tracking-[0.1em]"
          style={{ color: tone }}
        >
          {eyebrow}
        </span>
        <h2 className="text-[15px] font-semibold tracking-tight text-(--color-text-primary)">{title}</h2>
        <span className="text-[12.5px] text-(--color-text-tertiary)">{note}</span>
      </div>
      {children}
    </section>
  )
}

/** The three pillars, stated as a cycle rather than a diagram, at this
    size the loop is a claim to make in six words, not a figure to draw. */
function LoopChips() {
  const pillars = [
    { label: "Identify", color: "var(--color-identify)" },
    { label: "Generate", color: "var(--color-generate)" },
    { label: "Defend", color: "var(--color-defend)" },
  ]
  return (
    <div className="flex items-center gap-1.5 text-[12px] text-(--color-text-tertiary)">
      {pillars.map((p, i) => (
        <span key={p.label} className="flex items-center gap-1.5">
          <span className="flex items-center gap-1.5 font-medium" style={{ color: p.color }}>
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: p.color }} />
            {p.label}
          </span>
          <span aria-hidden>{i < pillars.length - 1 ? "→" : "↻"}</span>
        </span>
      ))}
    </div>
  )
}

function Figure({
  criterion,
  to,
  value,
  decimals,
  suffix = "",
  caption,
  foot,
}: {
  criterion: string
  to: string
  value: number | null
  decimals: number
  suffix?: string
  caption: string
  foot?: ReactNode
}) {
  return (
    <Link
      to={to}
      className="group flex flex-col bg-(--color-surface-0) px-4 py-4 transition-colors hover:bg-(--color-surface-1) focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-(--color-focus)"
    >
      <span className="text-[10.5px] font-semibold uppercase tracking-[0.1em] text-(--color-text-tertiary)">
        {criterion}
      </span>
      <span className="mt-2 text-[30px] font-semibold leading-none tracking-tight text-(--color-text-primary)">
        <CountUpValue target={value} decimals={decimals} suffix={suffix} />
      </span>
      <span className="mt-2 text-[13px] font-medium text-(--color-text-secondary)">{caption}</span>
      {foot && <span className="mt-0.5 text-[12px] leading-snug text-(--color-text-tertiary)">{foot}</span>}
      <span className="mt-2.5 text-[12px] font-medium text-(--color-text-tertiary) transition-colors group-hover:text-(--color-attack)">
        Evidence →
      </span>
    </Link>
  )
}
