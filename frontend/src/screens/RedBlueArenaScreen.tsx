/**
 * Hardening: the closed loop, measured.
 *
 * The simulator used to lead this screen; it is the Live screen's job
 * now, and running the same panel twice was duplication rather than
 * emphasis. What is left is the aggregate result the loop produces.
 *
 * The two arms are plotted separately, side by side, because they measure
 * genuinely different things: a red-team bypass rate against a payment
 * agent, and a black-box evasion rate against a tabular model. Overlaying
 * them on shared axes invited reading them as comparable, which they are
 * not.
 */

import { useMemo } from "react"
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { useApiData } from "../api/hooks"
import type { AgenticMeta, ClosedLoopResult } from "../api/types"
import { Skeleton } from "../components/ui/Skeleton"
import { TECHNIQUES } from "../components/LiveSandbox"
import { PageHeader } from "../components/ui/PageHeader"

const TOOLTIP_STYLE = {
  background: "var(--color-surface-0)",
  border: "1px solid var(--color-border)",
  borderRadius: 8,
  fontSize: 12,
}

export function RedBlueArenaScreen() {
  const { data: loop, loading } = useApiData<ClosedLoopResult>("/api/orchestrate/closed-loop")

  const agenticData = useMemo(
    () =>
      (loop?.agentic_rounds ?? []).map((r) => ({
        round: `R${r.round_num}`,
        bypass: r.overall_bypass_rate === null ? null : Math.round(r.overall_bypass_rate * 1000) / 10,
      })),
    [loop],
  )

  const tabularData = useMemo(
    () =>
      (loop?.tabular_adversarial?.rounds ?? []).map((r) => ({
        round: `R${r.round}`,
        evasion: Math.round(r.evasion_rate * 1000) / 10,
        prauc: Math.round(r.clean_eval.pr_auc * 1000) / 10,
        sigma: r.mean_perturbation_std_units ?? null,
      })),
    [loop],
  )

  const sigmas = tabularData.map((d) => d.sigma).filter((v): v is number => v !== null)
  const sigmaFirst = sigmas.length ? sigmas[0] : null
  const sigmaLast = sigmas.length ? sigmas[sigmas.length - 1] : null

  return (
    <div className="mx-auto flex max-w-[1240px] flex-col gap-6 px-4 py-6 sm:px-6">
      <PageHeader
        pillar="The loop"
        color="var(--color-attack)"
        title="Hardening"
        lede="Each round, whatever got through is folded back into the defense's training set and the attacker tries again."
      />

      {loading && <Skeleton className="h-72" />}

      {!loading && !loop && (
        <p className="rounded-(--radius-panel) border border-dashed border-(--color-border) px-4 py-8 text-center text-[13px] text-(--color-text-tertiary)">
          The closed loop has not been run yet, run{" "}
          <code className="font-mono text-[12px]">python -m janus.orchestrate.persist closed_loop</code>
        </p>
      )}

      {loop && (
        <>
          {loop.agentic_meta && <Provenance meta={loop.agentic_meta} />}

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <ChartPanel
              title="Payment-agent attacks"
              note="Round 0 runs undefended. From round 1 the Mandate Firewall is inline and retrains on every payload that succeeded in the round before."
              legend={[{ label: "Attacks that succeeded", color: "var(--color-attack)" }]}
            >
              <LineChart data={agenticData} margin={{ top: 8, right: 18, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-hairline)" vertical={false} />
                <XAxis dataKey="round" stroke="var(--color-text-tertiary)" fontSize={11} tickLine={false} />
                <YAxis
                  stroke="var(--color-text-tertiary)"
                  fontSize={11}
                  unit="%"
                  domain={[0, 100]}
                  tickLine={false}
                  width={42}
                />
                <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => [`${v}%`, "Succeeded"]} />
                <Line
                  type="monotone"
                  dataKey="bypass"
                  name="Attacks that succeeded"
                  stroke="var(--color-attack)"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  connectNulls
                />
              </LineChart>
            </ChartPanel>

            <ChartPanel
              title="Model evasion"
              note="A black-box attacker perturbs known fraud until the model stops flagging it. Each round's successful evasions go into training, then it tries again."
              legend={[
                { label: "Evaded detection", color: "var(--color-attack)" },
                { label: "Accuracy on ordinary traffic", color: "var(--color-defense)" },
              ]}
            >
              <LineChart data={tabularData} margin={{ top: 8, right: 18, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-hairline)" vertical={false} />
                <XAxis dataKey="round" stroke="var(--color-text-tertiary)" fontSize={11} tickLine={false} />
                <YAxis
                  stroke="var(--color-text-tertiary)"
                  fontSize={11}
                  unit="%"
                  domain={[0, 100]}
                  tickLine={false}
                  width={42}
                />
                <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v, name) => [`${v}%`, name]} />
                <Line
                  type="monotone"
                  dataKey="evasion"
                  name="Evaded detection"
                  stroke="var(--color-attack)"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="prauc"
                  name="Accuracy on ordinary traffic"
                  stroke="var(--color-defense)"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ChartPanel>
          </div>

          <p className="text-[12.5px] leading-relaxed text-(--color-text-tertiary)">
            The red lines falling is the result; the green line staying flat is the check on it. Resistance is
            not being bought by getting worse at ordinary traffic.
            {sigmaFirst !== null && sigmaLast !== null && (
              <>
                {" "}
                The other half of the story is what evasion costs the attacker: the transactions that still get
                through have been displaced{" "}
                <span className="font-semibold text-(--color-text-secondary)">
                  {sigmaFirst.toFixed(1)}σ → {sigmaLast.toFixed(1)}σ
                </span>{" "}
                from normal across the rounds, so what survives is progressively less like a plausible payment.
              </>
            )}
          </p>

          <TechniqueBreakdown loop={loop} />
        </>
      )}
    </div>
  )
}

/** Who actually ran the attacks. A red-team result is only as interesting
 *  as the adversary behind it, and "an adaptive frontier model with the
 *  full history of what got caught" and "a library of four templates" are
 *  very different claims. */
function Provenance({ meta }: { meta: AgenticMeta }) {
  const live = meta.backend === "openai" || meta.backend === "local"
  const redTeamCalls = meta.llm_call_summary?.red_team?.calls
  const agentCalls = meta.llm_call_summary?.shopping_agent?.calls

  return (
    <section className="rounded-(--radius-panel) border border-(--color-hairline) px-4 py-3">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1.5 text-[12.5px]">
        <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-(--color-text-tertiary)">
          Adversary
        </span>
        {live && meta.red_team_model ? (
          <span className="text-(--color-text-secondary)">
            <span className="font-semibold text-(--color-text-primary)">{meta.red_team_model}</span> writing each
            payload from the full history of what was caught, against a{" "}
            <span className="font-semibold text-(--color-text-primary)">{meta.shopping_agent_model}</span> shopping
            agent
          </span>
        ) : (
          <span className="text-(--color-text-secondary)">
            Deterministic template library: templated variation, not adaptive reasoning
          </span>
        )}
        <span className="ml-auto flex flex-wrap gap-x-4 gap-y-1 tabular-nums text-(--color-text-tertiary)">
          <span>{meta.total_attempts} attempts</span>
          {redTeamCalls !== undefined && agentCalls !== undefined && (
            <span>
              {redTeamCalls + agentCalls} model calls
            </span>
          )}
          {meta.wall_clock_s !== null && <span>{Math.round(meta.wall_clock_s / 60)} min</span>}
        </span>
      </div>
      {meta.template_fallbacks > 0 && (
        <p className="mt-1.5 text-[11.5px] leading-relaxed text-(--color-text-tertiary)">
          {meta.template_fallbacks} of {meta.total_attempts} payload generations were refused by the provider&rsquo;s
          own safety classifier and fell back to the template library. Those attempts are counted in the rates
          above and are not model-authored.
        </p>
      )}
    </section>
  )
}

function ChartPanel({
  title,
  note,
  legend,
  children,
}: {
  title: string
  note: string
  legend: { label: string; color: string }[]
  children: React.ReactElement
}) {
  return (
    <section className="rounded-(--radius-panel) border border-(--color-hairline)">
      <header className="border-b border-(--color-hairline) px-4 py-3">
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <h2 className="text-[14px] font-semibold text-(--color-text-primary)">{title}</h2>
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            {legend.map((l) => (
              <span key={l.label} className="inline-flex items-center gap-1.5 text-[11px] text-(--color-text-tertiary)">
                <span className="h-1.5 w-1.5 rounded-full" style={{ background: l.color }} />
                {l.label}
              </span>
            ))}
          </div>
        </div>
        <p className="mt-1 text-[12px] leading-relaxed text-(--color-text-tertiary)">{note}</p>
      </header>
      <div className="px-2 py-3">
        <ResponsiveContainer width="100%" height={220}>
          {children}
        </ResponsiveContainer>
      </div>
    </section>
  )
}

/**
 * Per technique: what it achieved with no defense inline, and the worst it
 * achieved in any defended round.
 *
 * The undefended column is not padding. An earlier version of this panel
 * carried a hand-written closing line naming one technique as the standing
 * weak point, and a later run made it false, with a stronger shopping
 * agent, two of the four techniques failed before the firewall was even
 * switched on, which means their defended zero says nothing about the
 * firewall. Both numbers are shown, and the summary is computed from them,
 * so the claim cannot drift from the measurement again.
 */
function TechniqueBreakdown({ loop }: { loop: ClosedLoopResult }) {
  const undefended = loop.agentic_rounds.find((r) => !r.firewall_present)
  const defended = loop.agentic_rounds.filter((r) => r.firewall_present)
  if (defended.length === 0) return null

  const rows = TECHNIQUES.map((t) => {
    const rates = defended
      .map((r) => r.technique_stats[t.id]?.bypass_rate)
      .filter((v): v is number => v !== null && v !== undefined)
    const base = undefended?.technique_stats[t.id]?.bypass_rate ?? null
    return {
      ...t,
      base,
      worst: rates.length ? Math.max(...rates) : null,
      // Only a technique that worked undefended can demonstrate anything
      // about the defense that stopped it.
      informative: base !== null && base > 0,
    }
  }).filter((r) => r.worst !== null)

  const closedByFirewall = rows.filter((r) => r.informative && r.worst === 0)
  const stillOpen = rows.filter((r) => (r.worst ?? 0) > 0)
  const neverWorked = rows.filter((r) => !r.informative)

  return (
    <section className="overflow-hidden rounded-(--radius-panel) border border-(--color-hairline)">
      <header className="border-b border-(--color-hairline) bg-(--color-surface-1) px-4 py-2.5">
        <h2 className="text-[13px] font-semibold text-(--color-text-primary)">Which attacks the defense closes</h2>
      </header>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[620px] text-left">
          <thead>
            <tr className="border-b border-(--color-hairline) text-[10.5px] uppercase tracking-[0.08em] text-(--color-text-tertiary)">
              <th className="py-2 pl-4 pr-3 font-semibold">Technique</th>
              <th className="py-2 pr-3 text-right font-semibold">Undefended</th>
              <th className="py-2 pr-3 text-right font-semibold">Firewall inline</th>
              <th className="py-2 pr-4 font-semibold">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-b border-(--color-hairline) align-top last:border-0">
                <td className="py-2.5 pl-4 pr-3">
                  <div className="text-[12.5px] font-medium text-(--color-text-primary)">{r.label}</div>
                  <div className="text-[11.5px] text-(--color-text-tertiary)">{r.goal}</div>
                </td>
                <td className="py-2.5 pr-3 text-right text-[12.5px] tabular-nums text-(--color-text-secondary)">
                  {r.base === null ? "n/a" : `${(r.base * 100).toFixed(0)}%`}
                </td>
                <td className="py-2.5 pr-3 text-right text-[12.5px] tabular-nums text-(--color-text-secondary)">
                  {r.worst === null ? "n/a" : `${(r.worst * 100).toFixed(0)}%`}
                </td>
                <td className="py-2.5 pr-4 text-[12px]">
                  {!r.informative ? (
                    <span className="text-(--color-text-tertiary)">Never landed, defended or not</span>
                  ) : r.worst === 0 ? (
                    <span className="font-semibold text-(--color-defense)">Closed by the firewall</span>
                  ) : (
                    <span className="font-semibold text-(--color-attack)">Still gets through</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="border-t border-(--color-hairline) px-4 py-3 text-[12.5px] leading-relaxed text-(--color-text-tertiary)">
        {closedByFirewall.length > 0 && (
          <>
            {closedByFirewall.length} of {rows.length} techniques worked against the undefended agent and were
            closed completely once the Mandate Firewall was inline. Each violates something structural that a
            deterministic check can test against the signed mandate: a quantity constraint, an account binding,
            a stated total.{" "}
          </>
        )}
        {neverWorked.length > 0 && (
          <>
            {neverWorked.length} never landed even undefended, so their defended zero is a property of the
            shopping agent&rsquo;s own instruction-following rather than evidence about the firewall, stated
            because the aggregate rate does not distinguish the two.{" "}
          </>
        )}
        {stillOpen.length > 0 && (
          <>
            <span className="font-semibold text-(--color-text-secondary)">
              {stillOpen.map((r) => r.label).join(", ")}
            </span>{" "}
            still gets through: nothing in a signed mandate says which of several permitted items is the better
            buy, so there is no structural ground truth to check and only the text classifier stands against it.
          </>
        )}
      </p>
    </section>
  )
}
