/**
 * Detection, the Defend pillar's evidence.
 *
 * Three things, in the order a judge needs them:
 *
 *   1. The six detector families side by side in one table, sorted the way
 *      they should be read, with the score as a bar so the comparison is
 *      visual rather than digit-by-digit. Caveats live in the row that
 *      earned them and are spelled out beneath, so a perfect score on a
 *      synthetic test set can never pass as a headline.
 *   2. The stack: what the members score alone, what they score combined,
 *      and where the combined decision actually routes traffic. The tier
 *      strip used to be a design explainer with no numbers behind it --
 *      the meta-learner was implemented but never run on data. It is now
 *      a measured distribution, and it says how much volume each tier
 *      takes and how much of the fraud it catches.
 *   3. Why a decline happened, and how long a decision takes; the two
 *      questions that decide whether any of this is deployable.
 */

import { useState } from "react"
import { useApiData } from "../api/hooks"
import type {
  EnsembleResult,
  EvalReport,
  Explanations,
  GbmMetrics,
  GnnHybridMetrics,
  LatencyBudget,
  LatencyProfile,
  MuleGeneralization,
  MuleRingMetrics,
  OnboardingResult,
  SequenceTransformerMetrics,
  TierRow,
  TimeAblation,
} from "../api/types"
import { Skeleton } from "../components/ui/Skeleton"
import { PageHeader } from "../components/ui/PageHeader"

interface Row {
  family: string
  catches: string
  report: EvalReport
  basis: string
  /** Footnote index into CAVEATS, when the number needs qualifying. */
  caveat?: number
  extra?: string
}

const MEMBER_LABEL: Record<string, string> = {
  gbm_tabular: "Gradient-boosted trees",
  gnn_graph: "Graph neural network",
  anomaly_isolation_forest: "Unsupervised anomaly",
}

const TIER_LABEL: Record<string, { label: string; color: string; desc: string }> = {
  auto_approve: { label: "Auto-approve", color: "var(--color-tier-approve)", desc: "Straight through." },
  step_up: { label: "Step-up auth", color: "var(--color-tier-stepup)", desc: "Challenge the cardholder." },
  review: { label: "Review", color: "var(--color-tier-review)", desc: "Queue for a human analyst." },
  decline: { label: "Decline", color: "var(--color-tier-decline)", desc: "Refuse the authorization." },
}

export function DefenseConsoleScreen() {
  const gbm = useApiData<GbmMetrics>("/api/defend/gbm")
  const gnn = useApiData<GnnHybridMetrics>("/api/defend/gnn")
  const mule = useApiData<MuleRingMetrics>("/api/generate/mule-ring")
  const muleGen = useApiData<MuleGeneralization>("/api/generate/mule-ring/generalization")
  const voice = useApiData<EvalReport>("/api/generate/voice-scam")
  const seq = useApiData<SequenceTransformerMetrics>("/api/defend/sequence")
  const onboarding = useApiData<OnboardingResult>("/api/generate/identity-onboarding")
  const ensemble = useApiData<EnsembleResult>("/api/defend/ensemble")
  const shap = useApiData<Explanations>("/api/defend/explanations")
  const profile = useApiData<LatencyProfile>("/api/defend/latency")
  const budget = useApiData<LatencyBudget>("/api/orchestrate/latency-budget")
  const ablation = useApiData<TimeAblation>("/api/defend/time-ablation")

  const loading = gbm.loading || gnn.loading || mule.loading || voice.loading || seq.loading

  const rows: Row[] = []
  if (gbm.data)
    rows.push({
      family: "Gradient-boosted trees",
      catches: "Fraud in ordinary card traffic",
      report: gbm.data,
      basis: `${gbm.data.n_rows.toLocaleString()} real transactions · ${gbm.data.n_fraud} fraudulent`,
      extra: `${gbm.data.inference_ms_per_1000_rows.toFixed(1)}ms per 1,000 rows`,
    })
  if (gnn.data)
    rows.push({
      family: "Graph neural network + trees",
      catches: "Rings that no single transaction reveals",
      report: gnn.data.hybrid,
      basis: `${gnn.data.n_nodes.toLocaleString()} entities · ${gnn.data.n_edges.toLocaleString()} relationships`,
      extra: `graph alone ${gnn.data.gnn_only.pr_auc.toFixed(2)} → combined ${gnn.data.hybrid.pr_auc.toFixed(2)}`,
    })
  if (onboarding.data)
    rows.push({
      family: "Onboarding application scorer",
      catches: "GenAI-assembled synthetic identities at account opening",
      report: onboarding.data,
      basis: `${onboarding.data.population.n_applications.toLocaleString()} applications · ${onboarding.data.population.n_synthetic} synthetic`,
      caveat: 3,
      extra: `${(onboarding.data.false_positive_rate_thin_file_legit * 100).toFixed(2)}% false-positive rate on genuine thin-file applicants`,
    })
  if (mule.data)
    rows.push({
      family: "Account-level graph features",
      catches: "Money-mule networks moving funds",
      report: mule.data,
      basis: `${mule.data.n_injected_rings} rings injected into real mobile-money data`,
      caveat: 0,
      extra: muleGen.data
        ? `holds at ${muleGen.data.out_of_distribution_subtle_style.pr_auc.toFixed(2)} on ring shapes it never trained on`
        : undefined,
    })
  if (seq.data)
    rows.push({
      family: "Sequence transformer",
      catches: "Automated card-testing runs",
      report: seq.data,
      basis: `${seq.data.n_test.toLocaleString()} simulated entity histories`,
      caveat: 1,
    })
  if (voice.data)
    rows.push({
      family: "Behavioural fingerprint",
      catches: "Victims talked into paying by a cloned voice",
      report: voice.data,
      basis: `${voice.data.n_total.toLocaleString()} transactions · ${voice.data.n_positive} scam payments`,
      caveat: 2,
    })

  const used = [...new Set(rows.map((r) => r.caveat).filter((c): c is number => c !== undefined))].sort()
  const tierViews = ensemble.data ? buildTierViews(ensemble.data) : null

  return (
    <div className="mx-auto flex max-w-[1240px] flex-col gap-7 px-4 py-6 sm:px-6">
      <PageHeader
        pillar="Defend"
        color="var(--color-defend)"
        title="Detection"
        lede="No single model covers fraud that arrives as a row, a sequence, a graph and a paragraph of text. Each family is measured on the substrate it is built for, then stacked into one decision."
      />

      {loading && <Skeleton className="h-64" />}

      {/* Two columns fold into the first one below md rather than hiding
          behind a horizontal scroll. Six columns cannot fit a phone, and a
          table a reader has to discover is scrollable is a table whose most
          important column is the only one they ever see. */}
      {!loading && (
        <div className="overflow-x-auto rounded-(--radius-panel) border border-(--color-hairline)">
          <table className="w-full text-left md:min-w-[860px]">
            <thead>
              <tr className="border-b border-(--color-hairline) bg-(--color-surface-1) text-[10.5px] uppercase tracking-[0.08em] text-(--color-text-tertiary)">
                <th className="py-2.5 pl-4 pr-3 font-semibold">Detector</th>
                <th className="hidden py-2.5 pr-3 font-semibold md:table-cell">Catches</th>
                <th className="py-2.5 pr-3 font-semibold md:w-[190px]">PR-AUC</th>
                <th className="py-2.5 pr-3 text-right font-semibold">Recall</th>
                <th className="hidden py-2.5 pr-3 text-right font-semibold sm:table-cell">Precision</th>
                <th className="hidden py-2.5 pr-4 font-semibold md:table-cell">Measured on</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.family} className="border-b border-(--color-hairline) align-top last:border-0">
                  <td className="py-3 pl-4 pr-3">
                    <div className="text-[13px] font-semibold text-(--color-text-primary)">{r.family}</div>
                    <div className="mt-0.5 text-[12px] text-(--color-text-secondary) md:hidden">{r.catches}</div>
                    {r.extra && <div className="mt-0.5 text-[11.5px] text-(--color-text-tertiary)">{r.extra}</div>}
                    <div className="mt-0.5 text-[11.5px] text-(--color-text-tertiary) md:hidden">
                      {r.basis}
                      {r.caveat !== undefined && (
                        <sup className="ml-1 font-semibold text-(--color-tier-review)">{r.caveat + 1}</sup>
                      )}
                    </div>
                  </td>
                  <td className="hidden py-3 pr-3 text-[12.5px] text-(--color-text-secondary) md:table-cell">
                    {r.catches}
                  </td>
                  <td className="py-3 pr-3">
                    <ScoreBar value={r.report.pr_auc} />
                  </td>
                  <td className="py-3 pr-3 text-right text-[12.5px] tabular-nums text-(--color-text-secondary)">
                    {(r.report.recall * 100).toFixed(0)}%
                  </td>
                  <td className="hidden py-3 pr-3 text-right text-[12.5px] tabular-nums text-(--color-text-secondary) sm:table-cell">
                    {(r.report.precision * 100).toFixed(0)}%
                  </td>
                  <td className="hidden py-3 pr-4 text-[12px] text-(--color-text-tertiary) md:table-cell">
                    {r.basis}
                    {r.caveat !== undefined && (
                      <sup className="ml-1 font-semibold text-(--color-tier-review)">{r.caveat + 1}</sup>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {used.length > 0 && (
        <ol className="flex flex-col gap-1.5">
          {used.map((i) => (
            <li key={i} className="flex gap-2 text-[12px] leading-relaxed text-(--color-text-tertiary)">
              <span className="shrink-0 font-semibold text-(--color-tier-review)">{i + 1}</span>
              {CAVEATS[i]}
            </li>
          ))}
        </ol>
      )}

      <p className="text-[12px] leading-relaxed text-(--color-text-tertiary)">
        Scores are PR-AUC because fraud is under 1% of traffic, plain accuracy is ~99% for a model that never
        flags anything.
      </p>

      {/* ---------------- the stack ---------------- */}
      {ensemble.data && (
        <section className="rounded-(--radius-panel) border border-(--color-hairline)">
          <header className="border-b border-(--color-hairline) px-4 py-3">
            <h2 className="text-[14px] font-semibold text-(--color-text-primary)">
              Three families, one decision
            </h2>
            <p className="mt-0.5 text-[12.5px] leading-relaxed text-(--color-text-tertiary)">
              {ensemble.data.split_note}
            </p>
          </header>

          <div className="grid gap-px bg-(--color-hairline) md:grid-cols-[1fr_auto]">
            <div className="bg-(--color-surface-0) px-4 py-3">
              <ul className="flex flex-col gap-2">
                {Object.entries(ensemble.data.members).map(([key, report]) => (
                  <li key={key} className="grid grid-cols-[1fr_170px] items-center gap-3">
                    <span className="text-[12.5px] text-(--color-text-secondary)">
                      {MEMBER_LABEL[key] ?? key}
                      <span className="ml-2 text-[11.5px] text-(--color-text-tertiary)">
                        weight {ensemble.data!.member_weights[key]?.toFixed(2) ?? ", "}
                      </span>
                    </span>
                    <ScoreBar value={report.pr_auc} muted />
                  </li>
                ))}
                <li className="mt-1 grid grid-cols-[1fr_170px] items-center gap-3 border-t border-(--color-hairline) pt-2.5">
                  <span className="text-[13px] font-semibold text-(--color-text-primary)">Stacked</span>
                  <ScoreBar value={ensemble.data.stacked.pr_auc} />
                </li>
              </ul>
            </div>

            <div className="flex min-w-[210px] flex-col justify-center bg-(--color-surface-0) px-4 py-3 text-[12px] text-(--color-text-tertiary)">
              <span>
                {ensemble.data.graph.n_nodes.toLocaleString()} nodes ·{" "}
                {ensemble.data.graph.n_edges.toLocaleString()} edges
              </span>
              <span>
                {ensemble.data.n_train.toLocaleString()} train / {ensemble.data.n_meta.toLocaleString()} meta /{" "}
                {ensemble.data.n_test.toLocaleString()} test
              </span>
              <span>
                trained on {ensemble.data.graph.device} in {Math.round(ensemble.data.wall_clock_s / 60)} min
              </span>
            </div>
          </div>

          {tierViews && <TierTables views={tierViews} />}
        </section>
      )}

      {/* ---------------- reasons and latency ---------------- */}
      <div className="grid gap-6 lg:grid-cols-2">
        {shap.data && (
          <section className="rounded-(--radius-panel) border border-(--color-hairline)">
            <header className="border-b border-(--color-hairline) px-4 py-3">
              <h2 className="text-[14px] font-semibold text-(--color-text-primary)">Why it declined</h2>
              <p className="mt-0.5 text-[12.5px] text-(--color-text-tertiary)">
                SHAP contributions for the highest-risk rows the fast path scored.
              </p>
            </header>
            <ul className="divide-y divide-(--color-hairline)">
              {shap.data.examples.slice(0, 3).map((ex, i) => (
                <li key={i} className="px-4 py-2.5">
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-[12px] text-(--color-text-tertiary)">
                      risk {ex.risk_score.toFixed(3)} · labelled {ex.true_label === 1 ? "fraud" : "legitimate"}
                    </span>
                  </div>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {ex.top_reasons.map((r) => (
                      <span
                        key={r.feature}
                        className="rounded-full px-2 py-0.5 font-mono text-[11px] tabular-nums"
                        style={{
                          color: r.shap >= 0 ? "var(--color-attack)" : "var(--color-defense)",
                          background: r.shap >= 0 ? "var(--color-attack-bg)" : "var(--color-defense-bg)",
                        }}
                        title={`${r.shap >= 0 ? "pushed toward" : "pushed away from"} fraud by ${Math.abs(r.shap).toFixed(3)} log-odds`}
                      >
                        {r.feature} {r.shap >= 0 ? "+" : ""}
                        {r.shap.toFixed(2)}
                      </span>
                    ))}
                  </div>
                </li>
              ))}
            </ul>
            <p className="border-t border-(--color-hairline) px-4 py-2.5 text-[11.5px] leading-relaxed text-(--color-text-tertiary)">
              {shap.data.note}
            </p>
          </section>
        )}

        <section className="rounded-(--radius-panel) border border-(--color-hairline)">
          <header className="border-b border-(--color-hairline) px-4 py-3">
            <h2 className="text-[14px] font-semibold text-(--color-text-primary)">Can it run inline?</h2>
            <p className="mt-0.5 text-[12.5px] text-(--color-text-tertiary)">
              Authorization is a synchronous decision. A model that cannot answer in time is not a control.
            </p>
          </header>

          {profile.data ? (
            <>
              <dl className="grid grid-cols-4 gap-px bg-(--color-hairline)">
                {(["p50", "p95", "p99", "max"] as const).map((k) => (
                  <div key={k} className="bg-(--color-surface-0) px-3 py-3">
                    <dt className="text-[10.5px] font-semibold uppercase tracking-[0.08em] text-(--color-text-tertiary)">
                      {k}
                    </dt>
                    <dd className="mt-1 text-[17px] font-semibold tabular-nums text-(--color-text-primary)">
                      {profile.data!.single_row_scoring_ms[k].toFixed(2)}
                      <span className="ml-0.5 text-[11px] font-normal text-(--color-text-tertiary)">ms</span>
                    </dd>
                  </div>
                ))}
              </dl>
              <p className="border-t border-(--color-hairline) px-4 py-2.5 text-[12px] leading-relaxed text-(--color-text-secondary)">
                Single-row scoring, {profile.data.single_row_scoring_ms.n_samples} samples on{" "}
                {profile.data.hardware.toLowerCase()}, against a{" "}
                {budget.data ? `${budget.data.fast_path_ms_budget}ms` : "300ms"} authorization budget.
              </p>
              <p className="border-t border-(--color-hairline) px-4 py-2.5 text-[11.5px] leading-relaxed text-(--color-text-tertiary)">
                {profile.data.note}
              </p>
            </>
          ) : (
            <p className="px-4 py-3 text-[12.5px] text-(--color-text-tertiary)">
              {budget.data?.note ?? "Latency profile not yet generated."}
            </p>
          )}
        </section>
      </div>

      {/* ---------------- the feature that was removed ---------------- */}
      {ablation.data && (
        <section className="rounded-(--radius-panel) border border-(--color-hairline) px-4 py-3">
          <h2 className="text-[14px] font-semibold text-(--color-text-primary)">
            A feature we removed on purpose
          </h2>
          <p className="mt-1 flex flex-wrap items-baseline gap-x-2 gap-y-1 text-[12.5px] text-(--color-text-secondary)">
            <span>
              With the capture clock:{" "}
              <span className="font-semibold tabular-nums text-(--color-text-primary)">
                {ablation.data.with_capture_clock.pr_auc.toFixed(4)}
              </span>{" "}
              PR-AUC.
            </span>
            <span>
              Without it:{" "}
              <span className="font-semibold tabular-nums text-(--color-text-primary)">
                {ablation.data.without_capture_clock.pr_auc.toFixed(4)}
              </span>
              .
            </span>
          </p>
          <p className="mt-1.5 text-[12px] leading-relaxed text-(--color-text-tertiary)">{ablation.data.note}</p>
        </section>
      )}
    </div>
  )
}

interface TierView {
  key: string
  label: string
  blurb: string
  rows: TierRow[]
}

/** Two views of the same scored population. Fixed probability cuts are what
 *  a regulator can be shown; capacity cuts are what an operations team can
 *  actually staff. Reporting only the first leaves an empty decline tier
 *  whenever the model is never confident enough to reach 0.85: true, and
 *  unreadable. Reporting only the second implies a confidence the model
 *  does not have. */
function buildTierViews(data: EnsembleResult): TierView[] {
  const views: TierView[] = [
    {
      key: "fixed",
      label: "Fixed probability cuts",
      blurb:
        "Decline above 0.85 calibrated risk, review above 0.60, challenge above 0.30. An empty tier here means the model is never that confident.",
      rows: data.tier_distribution.tiers,
    },
  ]
  if (data.tier_distribution_capacity) {
    views.push({
      key: "capacity",
      label: "Capacity-planned cuts",
      blurb:
        "Cuts placed so the top 0.2% of volume declines, the next 1% goes to review and the next 3% is challenged; the way a queue with a fixed number of analysts is actually sized.",
      rows: data.tier_distribution_capacity.tiers,
    })
  }
  return views
}

function TierTables({ views }: { views: TierView[] }) {
  const [active, setActive] = useState(views[views.length - 1].key)
  const view = views.find((v) => v.key === active) ?? views[0]

  return (
    <div className="border-t border-(--color-hairline)">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 px-4 py-2.5">
        <span className="text-[12.5px] font-medium text-(--color-text-secondary)">Where decisions go</span>
        {views.length > 1 && (
          <div className="flex gap-1" role="group" aria-label="Threshold basis">
            {views.map((v) => (
              <button
                key={v.key}
                type="button"
                onClick={() => setActive(v.key)}
                aria-pressed={v.key === active}
                className={
                  v.key === active
                    ? "inline-flex min-h-[34px] items-center rounded-full bg-(--color-text-primary) px-3 text-[11.5px] font-medium text-white sm:min-h-[24px]"
                    : "inline-flex min-h-[34px] items-center rounded-full px-3 text-[11.5px] font-medium text-(--color-text-tertiary) transition-colors hover:bg-(--color-surface-2) sm:min-h-[24px]"
                }
              >
                {v.label}
              </button>
            ))}
          </div>
        )}
      </div>
      <p className="px-4 pb-2.5 text-[11.5px] leading-relaxed text-(--color-text-tertiary)">{view.blurb}</p>

      <div className="overflow-x-auto border-t border-(--color-hairline)">
        <table className="w-full text-left sm:min-w-[560px]">
          <thead>
            <tr className="border-b border-(--color-hairline) bg-(--color-surface-1) text-[10.5px] uppercase tracking-[0.08em] text-(--color-text-tertiary)">
              <th className="py-2 pl-4 pr-3 font-semibold">Decision</th>
              <th className="py-2 pr-3 text-right font-semibold">Share of volume</th>
              <th className="py-2 pr-3 text-right font-semibold">Fraud caught</th>
              <th className="py-2 pr-4 text-right font-semibold">Of what lands here, fraud</th>
            </tr>
          </thead>
          <tbody>
            {view.rows.map((t) => {
              const meta = TIER_LABEL[t.tier]
              return (
                <tr key={t.tier} className="border-b border-(--color-hairline) last:border-0">
                  <td className="py-2 pl-4 pr-3">
                    <span className="flex items-center gap-1.5 text-[12.5px] font-medium text-(--color-text-primary)">
                      <span className="h-2 w-2 rounded-full" style={{ background: meta.color }} />
                      {meta.label}
                    </span>
                    <span className="ml-3.5 text-[11.5px] text-(--color-text-tertiary)">{meta.desc}</span>
                  </td>
                  <td className="py-2 pr-3 text-right text-[12.5px] tabular-nums text-(--color-text-secondary)">
                    {(t.share_of_volume * 100).toFixed(2)}%
                  </td>
                  <td className="py-2 pr-3 text-right text-[12.5px] tabular-nums text-(--color-text-secondary)">
                    {(t.share_of_fraud_caught * 100).toFixed(1)}%
                  </td>
                  <td className="hidden py-2 pr-4 text-right text-[12.5px] tabular-nums text-(--color-text-secondary) sm:table-cell">
                    {(t.fraud_rate_within_tier * 100).toFixed(2)}%
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const CAVEATS = [
  "Mule-ring detection is scored against rings this project injected into real mobile-money data, not against confirmed real mule accounts; the topology is realistic, the labels are ours.",
  "A perfect score on entirely synthetic sequences. This demonstrates that the architecture works; it is not a real-world detection rate. Only the first two rows are measured on real, labelled data.",
  "The hardest problem here, and the number says so. The payment is genuinely authorised by the real customer, so the only signal is that it does not look like their normal behaviour. Reported as measured rather than tuned upward, and it is why this signal routes to review rather than decline.",
  "The onboarding population is fully synthetic on both sides, because no public dataset labels synthetic-identity applications. What makes the number meaningful is the control group: it deliberately contains genuine thin-file applicants who share the fraud population's headline signature, and rings that rotate their own infrastructure. The per-sophistication split on the Simulation screen shows where detection actually falls off.",
]

/** The score as a bar as well as a figure, so families can be compared
    down the column at a glance instead of by reading digits. */
function ScoreBar({ value, muted = false }: { value: number; muted?: boolean }) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  const color = muted
    ? "var(--color-surface-4)"
    : value >= 0.8
      ? "var(--color-defense)"
      : value >= 0.5
        ? "var(--color-tier-review)"
        : "var(--color-attack)"
  return (
    <div className="flex items-center gap-2.5">
      <span
        className={`w-[46px] shrink-0 text-[15px] tabular-nums ${muted ? "font-medium text-(--color-text-secondary)" : "font-semibold text-(--color-text-primary)"}`}
      >
        {value.toFixed(2)}
      </span>
      <span className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-(--color-surface-3)">
        <span
          className="block h-full rounded-full transition-[width] duration-700 ease-out"
          style={{ width: `${pct}%`, background: color }}
        />
      </span>
    </div>
  )
}
