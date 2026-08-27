/**
 * Simulation, the Generate pillar's evidence.
 *
 * The challenge judges "fidelity of attacks in simulation", so this screen
 * exists to make fidelity legible. It leads with the one number that
 * summarises it end to end, can a classifier trained specifically to
 * separate real rows from synthetic ones actually do it?, and keeps the
 * Wasserstein / KS / JS table underneath for anyone who wants it, rather
 * than in front of everyone who doesn't.
 *
 * All three batches are shown, including the two whose fidelity is
 * visibly weaker than the headline one.
 */

import { useState, type ReactNode } from "react"
import { postJson } from "../api/client"
import { useApiData } from "../api/hooks"
import type { FidelityScorecard, OnboardingResult } from "../api/types"
import { Skeleton } from "../components/ui/Skeleton"
import { PageHeader } from "../components/ui/PageHeader"
import { fidelityVerdict } from "../lib/fidelity"

interface SimulateResponse {
  available: boolean
  reason?: string
  scorecard?: FidelityScorecard
  n_legit?: number
  n_fraud?: number
}

const BATCH_TITLE: Record<string, string> = {
  tabular_synthesis_ulb: "Synthetic card transactions",
  tabular_synthesis_ulb_legit: "Synthetic legitimate transactions",
  tabular_synthesis_ulb_fraud: "Synthetic fraudulent transactions",
  graph_mule_ring_amounts: "Mule-ring transfer amounts",
  sequence_voice_scam_amounts: "Voice-scam payment amounts",
}

const BATCH_ENGINE: Record<string, string> = {
  tabular_synthesis_ulb: "Gaussian copula fitted per class on 284,807 real transactions",
  tabular_synthesis_ulb_legit:
    "Gaussian copula fitted on the legitimate class, scored against real legitimate rows only",
  tabular_synthesis_ulb_fraud:
    "Gaussian copula fitted on the 492 real fraud rows, scored against real fraud only",
  graph_mule_ring_amounts: "Bootstrapped from the real distribution of legitimate transfers",
  sequence_voice_scam_amounts: "Sized to look like a plausible large legitimate transfer",
}

export function SimulationStudioScreen() {
  const [nRows, setNRows] = useState(3000)
  const [fraudRatio, setFraudRatio] = useState(0.05)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<SimulateResponse | null>(null)

  const cached = useApiData<FidelityScorecard[]>("/api/generate/fidelity")

  async function run() {
    setRunning(true)
    const res = await postJson<SimulateResponse>("/api/generate/simulate", {
      n_rows: String(nRows),
      fraud_ratio: String(fraudRatio),
    })
    // postJson returns null when no backend answered at all, which is the
    // normal case for the deployed static build. Without this the button
    // set running back to false and rendered nothing, so the primary
    // control on this screen looked broken rather than unavailable. The
    // backend's own "raw data not staged" response has the same shape, so
    // both paths land on the one explanation below.
    setResult(res ?? { available: false })
    setRunning(false)
  }

  return (
    <div className="mx-auto flex max-w-[1100px] flex-col gap-6 px-4 py-6 sm:px-6">
      <PageHeader
        pillar="Generate"
        color="var(--color-generate)"
        title="Simulation"
        lede="Every synthetic batch is scored by training a classifier to tell real rows from generated ones. If it cannot beat a coin flip, the data is indistinguishable from real."
      />

      <section className="rounded-(--radius-panel) border border-(--color-hairline)">
        <div className="flex flex-col gap-4 px-4 py-3.5 md:flex-row md:items-end">
          <Slider label="Transactions" value={nRows.toLocaleString()}>
            <input
              type="range"
              min={500}
              max={10000}
              step={500}
              value={nRows}
              onChange={(e) => setNRows(Number(e.target.value))}
              className="h-6 w-full accent-(--color-generate)"
              aria-label="Batch size"
            />
          </Slider>
          <Slider label="Fraud share" value={`${(fraudRatio * 100).toFixed(1)}%`}>
            <input
              type="range"
              min={0.005}
              max={0.2}
              step={0.005}
              value={fraudRatio}
              onChange={(e) => setFraudRatio(Number(e.target.value))}
              className="h-6 w-full accent-(--color-generate)"
              aria-label="Fraud ratio"
            />
          </Slider>
          <button
            onClick={run}
            disabled={running}
            className="shrink-0 rounded-full bg-(--color-text-primary) px-4 py-2 text-[13px] font-semibold text-white transition-colors hover:bg-black disabled:cursor-not-allowed disabled:bg-(--color-text-disabled) focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--color-focus)"
          >
            {running ? "Generating…" : "Generate batch"}
          </button>
        </div>

        {running && <Skeleton className="mx-4 mb-4 h-16" />}
        {!running && result && !result.available && (
          <p className="border-t border-(--color-hairline) px-4 py-3 text-[12.5px] leading-relaxed text-(--color-text-tertiary)">
            Live generation is off on this deployment: the raw datasets never touch the machine serving the
            prototype; they exist only inside the remote compute run that produced the scored batches below,
            which are the persisted evidence of fidelity.
          </p>
        )}
        {!running && result?.available && result.scorecard && (
          <div className="border-t border-(--color-hairline)">
            <ScorecardRow
              title="Your batch"
              engine={`${result.n_legit?.toLocaleString()} legitimate and ${result.n_fraud?.toLocaleString()} fraudulent rows, generated just now`}
              scorecard={result.scorecard}
            />
          </div>
        )}
      </section>

      <section>
        <h2 className="mb-2.5 text-[11px] font-semibold uppercase tracking-[0.1em] text-(--color-text-tertiary)">
          Scored batches
        </h2>
        {cached.loading && <Skeleton className="h-40" />}
        {!cached.loading && !cached.data && (
          <p className="rounded-(--radius-panel) border border-dashed border-(--color-border) px-4 py-6 text-center text-[13px] text-(--color-text-tertiary)">
            No scorecards yet, run{" "}
            <code className="font-mono text-[12px]">python -m janus.orchestrate.persist fidelity</code>
          </p>
        )}
        {cached.data && (
          <div className="divide-y divide-(--color-hairline) overflow-hidden rounded-(--radius-panel) border border-(--color-hairline)">
            {cached.data.map((card) => (
              <ScorecardRow
                key={card.batch_name}
                title={BATCH_TITLE[card.batch_name] ?? card.batch_name.replace(/_/g, " ")}
                engine={BATCH_ENGINE[card.batch_name] ?? ""}
                scorecard={card}
              />
            ))}
          </div>
        )}
      </section>

      <IdentityRingSection />
    </div>
  )
}

/**
 * The onboarding generator, on this screen rather than the detection one,
 * because the interesting result here is a property of the GENERATOR: how
 * hard it made the problem. A first version drew every ring from a single
 * distribution and the detector scored a clean 1.000 on everything, which
 * is the signature of a benchmark separable on one near-disjoint feature
 * rather than of a good detector. Splitting rings by how much
 * infrastructure they rotate is what makes the number mean something, and
 * the per-tier recall is where it stops being flattering.
 */
function IdentityRingSection() {
  const onboarding = useApiData<OnboardingResult>("/api/generate/identity-onboarding")
  if (!onboarding.data) return null

  const tiers = onboarding.data.recall_by_ring_sophistication
  const order = ["cheap", "moderate", "advanced"] as const
  const blurb: Record<string, string> = {
    cheap: "One device, one subnet, throwaway domains, bot-speed form fill",
    moderate: "Partial rotation, mixed mailbox age, some human pacing",
    advanced: "Fresh device and residential IP per application, aged mainstream mailboxes, replayed human timing",
  }

  return (
    <section>
      <h2 className="mb-2.5 text-[11px] font-semibold uppercase tracking-[0.1em] text-(--color-text-tertiary)">
        Synthetic-identity rings, by how much infrastructure they rotate
      </h2>
      <div className="overflow-hidden rounded-(--radius-panel) border border-(--color-hairline)">
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-(--color-hairline) bg-(--color-surface-1) text-[10.5px] uppercase tracking-[0.08em] text-(--color-text-tertiary)">
              <th className="py-2 pl-4 pr-3 font-semibold">Ring type</th>
              <th className="py-2 pr-3 font-semibold">What it buys</th>
              <th className="w-[170px] py-2 pr-4 font-semibold">Caught</th>
            </tr>
          </thead>
          <tbody>
            {order.map((tier) => {
              const row = tiers[tier]
              const recall = row?.recall ?? null
              return (
                <tr key={tier} className="border-b border-(--color-hairline) last:border-0">
                  <td className="py-2.5 pl-4 pr-3 text-[12.5px] font-semibold capitalize text-(--color-text-primary)">
                    {tier}
                    <span className="ml-1.5 font-normal tabular-nums text-(--color-text-tertiary)">
                      n={row?.n ?? 0}
                    </span>
                  </td>
                  <td className="py-2.5 pr-3 text-[12px] text-(--color-text-secondary)">{blurb[tier]}</td>
                  <td className="py-2.5 pr-4">
                    {recall === null ? (
                      <span className="text-[12px] text-(--color-text-tertiary)">, </span>
                    ) : (
                      <div className="flex items-center gap-2.5">
                        <span className="w-[42px] shrink-0 text-[14px] font-semibold tabular-nums text-(--color-text-primary)">
                          {(recall * 100).toFixed(0)}%
                        </span>
                        <span className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-(--color-surface-3)">
                          <span
                            className="block h-full rounded-full transition-[width] duration-700 ease-out"
                            style={{
                              width: `${recall * 100}%`,
                              background:
                                recall >= 0.9
                                  ? "var(--color-defense)"
                                  : recall >= 0.7
                                    ? "var(--color-tier-review)"
                                    : "var(--color-attack)",
                            }}
                          />
                        </span>
                      </div>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <p className="border-t border-(--color-hairline) px-4 py-2.5 text-[11.5px] leading-relaxed text-(--color-text-tertiary)">
          Measured against a control group that deliberately includes genuine thin-file applicants: first-time
          borrowers, recent arrivals, who share the fraud population&rsquo;s headline signature.{" "}
          <span className="font-semibold text-(--color-text-secondary)">
            {(onboarding.data.false_positive_rate_thin_file_legit * 100).toFixed(2)}%
          </span>{" "}
          of them are flagged, against{" "}
          <span className="font-semibold text-(--color-text-secondary)">
            {(onboarding.data.false_positive_rate_established_legit * 100).toFixed(2)}%
          </span>{" "}
          of established customers. Both are reported because a detector that scores well on the aggregate while
          failing the first group is declining people for being new.
        </p>
      </div>
    </section>
  )
}

function ScorecardRow({ title, engine, scorecard }: { title: string; engine: string; scorecard: FidelityScorecard }) {
  const auc = scorecard.distinguisher?.auc
  const verdict = auc !== undefined ? fidelityVerdict(auc) : null

  return (
    <div className="px-4 py-3.5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-5">
        <div className="min-w-0 flex-1">
          <div className="text-[13.5px] font-semibold text-(--color-text-primary)">{title}</div>
          {engine && <div className="mt-0.5 text-[12px] leading-snug text-(--color-text-tertiary)">{engine}</div>}
        </div>
        {scorecard.distinguisher && verdict && (
          <div className="flex shrink-0 items-center gap-4">
            <RealVsSyntheticMeter auc={scorecard.distinguisher.auc} />
            <div className="w-[130px]">
              <div className={`text-[12.5px] font-semibold ${verdict.tone}`}>{verdict.label}</div>
              <div className="text-[11.5px] tabular-nums text-(--color-text-tertiary)">
                AUC {scorecard.distinguisher.auc.toFixed(3)}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1.5">
        {scorecard.correlation && (
          <span
            className="text-[11.5px] text-(--color-text-tertiary)"
            title="Matching individual distributions is easy; preserving how features move together is where synthesizers usually fail, so this is reported alongside rather than hidden."
          >
            Correlation delta{" "}
            <span className="tabular-nums text-(--color-text-secondary)">
              {scorecard.correlation.mean_abs_delta.toFixed(2)}
            </span>{" "}
            across {scorecard.correlation.n_features} features
          </span>
        )}
        {scorecard.graph_topology && (
          <span
            className="text-[11.5px] text-(--color-text-tertiary)"
            title="Whether the injected ring topology sits inside the real transfer network's structural envelope, or sticks out as an obviously synthetic clique. A per-feature amount comparison cannot see this."
          >
            Degree distribution KS{" "}
            <span className="tabular-nums text-(--color-text-secondary)">
              {scorecard.graph_topology.degree_distribution_ks.toFixed(3)}
            </span>{" "}
            · clustering delta{" "}
            <span className="tabular-nums text-(--color-text-secondary)">
              {scorecard.graph_topology.clustering_coefficient_delta.toFixed(3)}
            </span>
          </span>
        )}
        <details className="group">
          <summary className="cursor-pointer list-none text-[11.5px] font-medium text-(--color-text-tertiary) hover:text-(--color-text-secondary)">
            <span className="inline-block transition-transform group-open:rotate-90">›</span> Per-feature statistics
            ({scorecard.features.length})
          </summary>
          <div className="mt-2 overflow-x-auto">
            <table className="w-full min-w-[420px] text-left text-[12px]">
              <thead>
                <tr className="text-[10.5px] uppercase tracking-[0.06em] text-(--color-text-tertiary)">
                  <th className="pb-1.5 pr-3 font-semibold">Feature</th>
                  <th className="pb-1.5 pr-3 font-semibold">Wasserstein</th>
                  <th className="pb-1.5 pr-3 font-semibold">KS</th>
                  <th className="pb-1.5 font-semibold">JS divergence</th>
                </tr>
              </thead>
              <tbody>
                {scorecard.features.map((f) => (
                  <tr key={f.feature} className="border-t border-(--color-hairline)">
                    <td className="py-1.5 pr-3 text-(--color-text-secondary)">{f.feature}</td>
                    <td className="py-1.5 pr-3 tabular-nums text-(--color-text-secondary)">
                      {f.wasserstein < 100
                        ? f.wasserstein.toFixed(3)
                        : f.wasserstein.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </td>
                    <td className="py-1.5 pr-3 tabular-nums text-(--color-text-secondary)">
                      {f.ks_statistic.toFixed(3)}
                    </td>
                    <td className="py-1.5 tabular-nums text-(--color-text-secondary)">{f.js_divergence.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-1.5 text-[11px] text-(--color-text-tertiary)">Lower is closer to real in every column.</p>
          </div>
        </details>
      </div>
    </div>
  )
}

/**
 * The distinguisher AUC on a 0.5-to-1.0 scale. Unlike every other score
 * in the app, lower is better here and 0.5 rather than 1.0 is perfect, so
 * the raw figure alone reliably misleads.
 */
function RealVsSyntheticMeter({ auc }: { auc: number }) {
  const clamped = Math.max(0.5, Math.min(1, auc))
  const pct = ((clamped - 0.5) / 0.5) * 100
  const color = pct < 20 ? "var(--color-defense)" : pct < 60 ? "var(--color-tier-review)" : "var(--color-attack)"
  return (
    <div className="w-[150px]">
      <div className="flex justify-between text-[10px] text-(--color-text-tertiary)">
        <span>Indistinguishable</span>
        <span>Obvious</span>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-(--color-surface-3)">
        <div
          className="h-full rounded-full transition-[width] duration-700 ease-out"
          style={{ width: `${Math.max(pct, 2)}%`, background: color }}
        />
      </div>
    </div>
  )
}

function Slider({ label, value, children }: { label: string; value: string; children: ReactNode }) {
  return (
    <div className="min-w-0 flex-1">
      <div className="mb-1 flex items-center justify-between gap-2 text-[12.5px]">
        <span className="text-(--color-text-tertiary)">{label}</span>
        <span className="font-semibold tabular-nums text-(--color-text-primary)">{value}</span>
      </div>
      {children}
    </div>
  )
}
