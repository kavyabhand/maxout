/**
 * The attack, drawn as a live swimlane simulation.
 *
 * This is the "attack in action" view the brief asks a prototype to show,
 * and it has been through two earlier shapes. The first streamed the
 * sandbox session as a scrolling monospace transcript, which forced the
 * viewer to parse tool names to work out where the defense intervened.
 * The second stacked eleven description cards, which was legible but so
 * tall and text-heavy that the run itself disappeared inside the copy.
 *
 * What is drawn now is the thing itself: three actors as lanes, the
 * eleven steps as nodes along a shared left-to-right time axis, and a
 * packet travelling the path between them. Where the packet stops IS the
 * result; a firewall node turning red with the trail ending there says
 * "stopped here" faster than any sentence can. Prose is reduced to one
 * line describing whichever step is executing.
 *
 * Pacing note: the backend resolves a scripted session in well under a
 * second and delivers every firewall verdict in a single final `result`
 * message, so an honest event stream would snap the whole diagram to its
 * end state instantly. `useStagedReveal` walks the already-computed
 * stages forward on a timer. That changes only the pacing of the
 * presentation, never the outcome; each stage's state is still derived
 * entirely from what the backend reported.
 */

import { useEffect, useMemo, useRef, useState } from "react"
import clsx from "clsx"
import type { SandboxMsg, Verdict } from "../../api/types"

export type StageState = "pending" | "active" | "passed" | "flagged" | "blocked" | "skipped"

export type Lane = "attacker" | "agent" | "firewall"

export interface FlowStage {
  key: string
  /** Node caption on the diagram. Kept to one or two words so eleven of
      them fit the axis without collision or rotation. */
  short: string
  label: string
  detail: string
  lane: Lane
  state: StageState
  notes?: string[]
  badgeOverride?: Partial<Record<StageState, string>>
}

const VERDICT_STATE: Record<Verdict, StageState> = {
  pass: "passed",
  flag: "flagged",
  block: "blocked",
}

interface BuildArgs {
  events: SandboxMsg[]
  firewallEnabled: boolean
  running: boolean
}

export function buildStages({ events, firewallEnabled, running }: BuildArgs): FlowStage[] {
  const payload = events.find((e) => e.type === "red_team_payload")
  const result = events.find((e) => e.type === "result")
  const transcript = events.filter((e) => e.type === "transcript_event")

  const calledTools = new Set<string>()
  for (const e of transcript) {
    if (e.type === "transcript_event" && e.tool_name) calledTools.add(e.tool_name)
  }

  const firewallByStage = new Map<string, { verdict: Verdict; reasons: string[] }>()
  if (result && result.type === "result") {
    for (const fe of result.firewall_events) {
      const existing = firewallByStage.get(fe.stage)
      // Keep the most severe verdict when a stage fires more than once --
      // the content scrubber runs on every product description read.
      const severity: Record<Verdict, number> = { pass: 0, flag: 1, block: 2 }
      if (!existing || severity[fe.verdict] >= severity[existing.verdict]) {
        firewallByStage.set(fe.stage, { verdict: fe.verdict, reasons: fe.reasons })
      }
    }
  }

  const started = events.length > 0
  const finished = result !== undefined

  /**
   * Takes a LIST of tools because the same logical step is reached by
   * different calls depending on which backend drives the agent: the
   * scripted policy returns product descriptions inline with
   * `search_catalog` and never calls `get_product`, whereas the LLM paths
   * call `get_product` separately.
   */
  const agentState = (tools: string[]): StageState => {
    if (!started) return "pending"
    if (tools.some((t) => calledTools.has(t))) return "passed"
    if (finished) return "skipped"
    return running ? "active" : "pending"
  }

  const firewallState = (backendStage: string): StageState => {
    if (!firewallEnabled) return "skipped"
    if (!finished) return started ? "active" : "pending"
    const hit = firewallByStage.get(backendStage)
    if (!hit) return "skipped"
    return VERDICT_STATE[hit.verdict]
  }

  const notesFor = (backendStage: string): string[] | undefined => {
    const hit = firewallByStage.get(backendStage)
    return hit && hit.reasons.length > 0 ? hit.reasons : undefined
  }

  return [
    {
      key: "craft",
      short: "Craft",
      label: "Craft payload",
      detail: "The red-team agent writes an injection tuned to this technique.",
      lane: "attacker",
      state: payload ? "passed" : started ? "active" : "pending",
    },
    {
      key: "poison",
      short: "Poison",
      label: "Poison listing",
      detail: "The payload is planted inside a merchant product description.",
      lane: "attacker",
      state: payload ? "passed" : "pending",
    },
    {
      key: "fw_scrub",
      short: "Screen text",
      label: "Screen listing text",
      detail: "An injection classifier inspects untrusted catalog text before the agent ever sees it.",
      lane: "firewall",
      state: firewallState("content_scrub"),
      notes: notesFor("content_scrub"),
      // A block here strips the payload and lets the session continue on
      // clean text rather than halting the run, so "Blocked" would misread.
      badgeOverride: { blocked: "Payload stripped" },
    },
    {
      key: "search",
      short: "Search",
      label: "Search catalog",
      detail: "The shopping agent looks for products matching the signed mandate.",
      lane: "agent",
      state: agentState(["search_catalog"]),
    },
    {
      key: "read",
      short: "Read",
      label: "Read listing",
      detail: "The agent takes in the description, including anything hidden inside it.",
      lane: "agent",
      state: agentState(["get_product", "search_catalog"]),
    },
    {
      key: "checkout",
      short: "Checkout",
      label: "Finalize checkout",
      detail: "The agent commits to a specific item and quantity.",
      lane: "agent",
      state: agentState(["finalize_checkout"]),
    },
    {
      key: "fw_checkout",
      short: "Constraints",
      label: "Constraint check",
      detail: "A maximal-flow test that every line item satisfies the signed mandate.",
      lane: "firewall",
      state: firewallState("checkout_constraints"),
      notes: notesFor("checkout_constraints"),
    },
    {
      key: "summary",
      short: "Confirm",
      label: "Confirm to user",
      detail: "The agent states in plain language what is about to be bought.",
      lane: "agent",
      state: agentState(["confirm_purchase_summary"]),
    },
    {
      key: "fw_summary",
      short: "Amount",
      label: "Amount check",
      detail: "The confirmation text must match the real total being charged.",
      lane: "firewall",
      state: firewallState("purchase_summary"),
      notes: notesFor("purchase_summary"),
    },
    {
      key: "credential",
      short: "Request pay",
      label: "Request payment",
      detail: "The agent asks the credential vault for a payment instrument.",
      lane: "agent",
      state: agentState(["request_payment_credential"]),
    },
    {
      key: "fw_account",
      short: "Binding",
      label: "Account binding",
      detail: "The credential request must name the session's own account, and no other.",
      lane: "firewall",
      state: firewallState("account_ref"),
      notes: notesFor("account_ref"),
    },
  ]
}

/**
 * Walks a reveal cursor forward once a run begins so the diagram plays
 * rather than snapping. Resets when a new run starts, detected by the
 * event list emptying.
 */
function useStagedReveal(total: number, started: boolean, stepMs: number): number {
  const [revealed, setRevealed] = useState(0)
  const timer = useRef<number | null>(null)

  useEffect(() => {
    if (!started) {
      setRevealed(0)
      return
    }
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches
    if (reduced) {
      setRevealed(total)
      return
    }
    setRevealed(0)
    let i = 0
    timer.current = window.setInterval(() => {
      i += 1
      setRevealed(i)
      if (i >= total && timer.current !== null) window.clearInterval(timer.current)
    }, stepMs)
    return () => {
      if (timer.current !== null) window.clearInterval(timer.current)
    }
  }, [started, total, stepMs])

  return revealed
}

export const STEP_MS = 300

const LANES: { id: Lane; label: string; color: string }[] = [
  { id: "attacker", label: "Red team", color: "var(--color-attack)" },
  { id: "agent", label: "Shopping agent", color: "#5b6472" },
  { id: "firewall", label: "Mandate Firewall", color: "var(--color-defense)" },
]

const STATE_FILL: Record<StageState, string> = {
  pending: "var(--color-surface-3)",
  active: "var(--color-flow)",
  passed: "var(--color-defense)",
  flagged: "var(--color-tier-review)",
  blocked: "var(--color-attack)",
  skipped: "var(--color-surface-3)",
}

const STATE_WORD: Record<StageState, string | null> = {
  pending: null,
  active: "Running",
  passed: "Passed",
  flagged: "Flagged",
  blocked: "Blocked",
  skipped: "Not reached",
}

// Diagram geometry, in viewBox units.
const VB_W = 1100
const VB_H = 208
const GUTTER = 132
const X0 = GUTTER + 34
const X1 = VB_W - 26
const LANE_Y: Record<Lane, number> = { attacker: 40, agent: 104, firewall: 168 }

export function AttackFlow({
  events,
  firewallEnabled,
  running,
}: {
  events: SandboxMsg[]
  firewallEnabled: boolean
  running: boolean
}) {
  const stages = useMemo(() => buildStages({ events, firewallEnabled, running }), [events, firewallEnabled, running])
  const started = events.length > 0
  const revealed = useStagedReveal(stages.length, started, STEP_MS)

  const shown = started ? Math.min(revealed, stages.length) : 0
  const playing = started && shown < stages.length

  // While the run plays, the detail line follows the packet. Once it
  // finishes it settles on the DECISIVE step rather than the final one:
  // an attack is often stopped early and then allowed to run harmlessly
  // to completion, and in that case resting on step 11 answers "what
  // happened last" when the question a viewer is actually asking is
  // "where did it stop".
  // The furthest step that actually executed. Not simply the last stage:
  // with the firewall off, the final three checkpoints never run, and
  // resting the packet and the caption on a step that was skipped implies
  // the run got somewhere it never went.
  let lastExecuted = 0
  stages.forEach((s, i) => {
    if (s.state === "passed" || s.state === "blocked" || s.state === "flagged") lastExecuted = i
  })
  const decisive = stages.findIndex((s) => s.state === "blocked" || s.state === "flagged")
  const packetIndex = playing ? Math.max(0, shown - 1) : lastExecuted
  // Once the run settles, the caption moves to the DECISIVE step rather
  // than the furthest one: an attack is often stopped early and then
  // allowed to run on harmlessly, and in that case "what happened last"
  // is not the question a viewer is asking.
  const focus = started ? stages[playing || decisive < 0 ? packetIndex : decisive] : null

  const step = (X1 - X0) / (stages.length - 1)
  const xy = (i: number): [number, number] => [X0 + i * step, LANE_Y[stages[i].lane]]

  // The packet rests on the last revealed node, always the furthest the
  // run actually reached, never the decisive step, so it is never shown
  // sitting on a checkpoint the payload went on to survive. CSS
  // transitions on cx/cy carry it along the segment, which is what gives
  // the diagram its sense of something travelling through the system.
  const [px, py] = xy(packetIndex)

  return (
    <div className="flex flex-col">
      <div className="flex items-center justify-between gap-4 px-4 pb-1 pt-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          {LANES.map((l) => (
            <span key={l.id} className="inline-flex items-center gap-1.5 text-[11px] text-(--color-text-tertiary)">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: l.color }} />
              {l.label}
            </span>
          ))}
        </div>
        <span className="shrink-0 text-[11px] tabular-nums text-(--color-text-tertiary)">
          {started ? `${playing ? "Running" : "Complete"} · ${shown}/${stages.length}` : `${stages.length} steps`}
        </span>
      </div>

      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          className="h-[208px] w-full min-w-[860px]"
          role="img"
          aria-label={`Attack pipeline, step ${shown} of ${stages.length}`}
        >
          {LANES.map((l) => (
            <g key={l.id}>
              <text
                x={GUTTER}
                y={LANE_Y[l.id] + 4}
                textAnchor="end"
                fontSize="11.5"
                fontWeight="600"
                fill={l.color}
                fontFamily="var(--font-sans)"
              >
                {l.label}
              </text>
              <line
                x1={X0 - 18}
                y1={LANE_Y[l.id]}
                x2={X1 + 14}
                y2={LANE_Y[l.id]}
                stroke="var(--color-hairline)"
                strokeWidth="1"
              />
            </g>
          ))}

          {/* One path element per hop, so a segment can be styled by whether
              the run has actually crossed it yet. */}
          {stages.slice(0, -1).map((s, i) => {
            const [ax, ay] = xy(i)
            const [bx, by] = xy(i + 1)
            const mid = (ax + bx) / 2
            const d = ay === by ? `M${ax} ${ay} L${bx} ${by}` : `M${ax} ${ay} C${mid} ${ay} ${mid} ${by} ${bx} ${by}`
            const crossed = i + 1 < shown
            const crossing = i + 1 === shown && playing
            // A hop into or out of a bypassed checkpoint stays an inert
            // hairline even once the run has passed it: with the firewall
            // off, a solid trail through a firewall node would imply a
            // check that never ran.
            const bypassed = stages[i].state === "skipped" || stages[i + 1].state === "skipped"
            const live = (crossed || crossing) && !bypassed
            return (
              <path
                key={s.key}
                d={d}
                fill="none"
                stroke={live ? "var(--color-flow)" : "var(--color-surface-3)"}
                strokeWidth={live ? 1.8 : 1.2}
                strokeDasharray={crossing && live ? "4 4" : bypassed ? "3 3" : undefined}
                className={crossing && live ? "animate-dash" : undefined}
              />
            )
          })}

          {stages.map((s, i) => {
            const [cx, cy] = xy(i)
            const isRevealed = i < shown
            const state: StageState = isRevealed ? s.state : "pending"
            const isCursor = isRevealed && i === packetIndex
            const dim = !isRevealed || state === "skipped"
            return (
              <g key={s.key}>
                {isCursor && playing && (
                  <circle cx={cx} cy={cy} r="7" fill={STATE_FILL[state]} className="animate-halo" />
                )}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isCursor ? 7.5 : 6}
                  fill={state === "skipped" ? "var(--color-surface-0)" : STATE_FILL[state]}
                  stroke={state === "skipped" ? "var(--color-surface-4)" : "var(--color-surface-0)"}
                  strokeWidth={state === "skipped" ? 1.5 : 2.5}
                  style={{ transition: "fill 200ms ease, r 200ms ease" }}
                />
                {state === "blocked" && (
                  <path
                    d={`M${cx - 2.6} ${cy - 2.6} L${cx + 2.6} ${cy + 2.6} M${cx + 2.6} ${cy - 2.6} L${cx - 2.6} ${cy + 2.6}`}
                    stroke="#fff"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                  />
                )}
                {state === "passed" && (
                  <path
                    d={`M${cx - 2.6} ${cy} L${cx - 0.6} ${cy + 2.2} L${cx + 2.8} ${cy - 2.4}`}
                    fill="none"
                    stroke="#fff"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                )}
                <text
                  x={cx}
                  y={cy + 24}
                  textAnchor="middle"
                  fontSize="10.5"
                  fontWeight={isCursor ? 700 : 500}
                  fill={dim ? "var(--color-text-disabled)" : "var(--color-text-secondary)"}
                  fontFamily="var(--font-sans)"
                >
                  {s.short}
                </text>
              </g>
            )
          })}

          {started && (
            <circle
              cx={px}
              cy={py}
              r="3"
              fill="var(--color-surface-0)"
              stroke="var(--color-flow)"
              strokeWidth="2"
              style={{ transition: `cx ${STEP_MS}ms linear, cy ${STEP_MS}ms linear` }}
            />
          )}
        </svg>
      </div>

      <StepDetail stage={focus} total={stages.length} />
    </div>
  )
}

/**
 * A single fixed-position line describing whichever step is executing.
 * Holding one slot rather than growing a list is what keeps the panel a
 * constant height while the run plays; the diagram above stays put
 * instead of being pushed down the page step by step.
 */
function StepDetail({ stage, total }: { stage: FlowStage | null; total: number }) {
  if (!stage) {
    return (
      <div className="border-t border-(--color-hairline) px-4 py-3 text-[13px] text-(--color-text-tertiary)">
        Choose a technique and run it; the payload travels the {total} steps above, and wherever it stops is
        the result.
      </div>
    )
  }

  const word = stage.badgeOverride?.[stage.state] ?? STATE_WORD[stage.state]
  const tone =
    stage.state === "blocked"
      ? "text-(--color-attack)"
      : stage.state === "flagged"
        ? "text-(--color-tier-review)"
        : stage.state === "passed"
          ? "text-(--color-defense)"
          : "text-(--color-text-tertiary)"

  return (
    <div key={stage.key} className="animate-rise border-t border-(--color-hairline) px-4 py-3">
      <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
        <span className="text-[13px] font-semibold text-(--color-text-primary)">{stage.label}</span>
        {word && <span className={clsx("text-[11px] font-semibold uppercase tracking-[0.06em]", tone)}>{word}</span>}
      </div>
      <p className="mt-0.5 text-[13px] leading-relaxed text-(--color-text-tertiary)">{stage.detail}</p>
      {stage.notes && stage.notes.length > 0 && (
        <ul className="mt-1.5 flex flex-col gap-0.5">
          {stage.notes.map((n) => (
            <li key={n} className="text-[12.5px] leading-relaxed text-(--color-text-secondary)">, {n}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
