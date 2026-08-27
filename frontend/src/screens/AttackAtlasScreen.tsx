/**
 * Attacks, the Identify pillar.
 *
 * The graph stays, because attacks, rails, channels and detectors really
 * are one connected structure and a list cannot show that. What went is
 * the copy around it: a three-line lede, an explainer paragraph under
 * each of the three coverage labels, and a subtitle on every control. The
 * status labels now explain themselves on hover, where a reader can ask
 * for the definition instead of being handed it three times.
 */

import { useEffect, useMemo, useState } from "react"
import clsx from "clsx"
import { useApiData } from "../api/hooks"
import type { AtlasGraph, AttackNode, CoverageSummary } from "../api/types"
import { ForceGraph } from "../components/ForceGraph"
import { StatusBadge } from "../components/ui/Badge"
import { Skeleton } from "../components/ui/Skeleton"
import { IdentifyAgentPanel } from "../components/IdentifyAgentPanel"
import { PageHeader } from "../components/ui/PageHeader"

const CATEGORY_LABELS: Record<string, string> = {
  A: "Identity & Onboarding",
  B: "Social Engineering",
  C: "Transaction & Network",
  D: "Agentic Commerce",
}

/** Node hues, matching ForceGraph's CATEGORY_COLOR map. */
const CATEGORY_SWATCH: Record<string, string> = {
  A: "#1f4b8e",
  B: "#f79e1b",
  C: "#ff5f00",
  D: "#eb001b",
}

const STATUS_TITLE: Record<string, string> = {
  simulated: "JANUS generates this attack for real and measures a detector against it.",
  modeled: "The attack's signals are represented in the system, but it is not generated end to end.",
  taxonomy_only: "Researched and mapped, but not simulated. Listed so breadth is not mistaken for depth.",
}

export function AttackAtlasScreen() {
  const { data: graph, loading: graphLoading } = useApiData<AtlasGraph>("/api/identify/atlas")
  const { data: coverage } = useApiData<CoverageSummary>("/api/identify/coverage")
  const [selected, setSelected] = useState<AttackNode | null>(null)
  const [autoSelected, setAutoSelected] = useState(false)
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null)
  const [view, setView] = useState<"map" | "list">("map")

  const attacks = useMemo(() => graph?.nodes.filter((n): n is AttackNode => n.kind === "attack") ?? [], [graph])

  // Land on the flagship agentic-commerce vector rather than an empty
  // inspector. Runs once, so a user's own selection is never overridden.
  useEffect(() => {
    if (autoSelected || attacks.length === 0) return
    const flagship = attacks.find((a) => a.id === "D14") ?? attacks.find((a) => a.status === "simulated")
    if (flagship) setSelected(flagship)
    setAutoSelected(true)
  }, [attacks, autoSelected])

  const visibleAttacks = categoryFilter ? attacks.filter((a) => a.category === categoryFilter) : attacks

  return (
    <div className="mx-auto flex max-w-[1400px] flex-col gap-5 px-4 py-6 sm:px-6">
      <PageHeader
        pillar="Identify"
        color="var(--color-identify)"
        title="Attack Atlas"
        lede="How generative AI is changing payment fraud, mapped to the rails it runs on and the detectors built for it here."
        action={coverage ? <CoverageBar coverage={coverage} /> : null}
      />

      {/* `items-start` matters: a stretched grid row would hand the graph
          column the height of the taller inspector column, and the graph is
          sized by its own box, not by its neighbour. */}
      <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[1fr_340px]">
        <div className="flex flex-col gap-4">
          {/* A definite height, not `flex-1` inside a viewport-locked
              column. That earlier layout only worked on a tall window: on a
              short one the grid handed this row ~150px, `min-h-[460px]` blew
              the section straight out of its track, and the graph painted
              over the inspector beneath it. A clamped viewport height keeps
              the map generous on a large screen and honest on a small one,
              and lets the page scroll like every other screen here. */}
          <section className="flex h-[clamp(360px,58vh,620px)] flex-col overflow-hidden rounded-(--radius-panel) border border-(--color-hairline)">
            <div className="flex flex-wrap items-center gap-2 border-b border-(--color-hairline) bg-(--color-surface-1) px-3 py-2">
              <div className="flex min-w-0 flex-wrap gap-1">
                {(["A", "B", "C", "D"] as const).map((c) => (
                  <button
                    key={c}
                    onClick={() => setCategoryFilter((prev) => (prev === c ? null : c))}
                    aria-pressed={categoryFilter === c}
                    title={CATEGORY_LABELS[c]}
                    className={clsx(
                      "inline-flex min-h-[30px] shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-1 text-[11.5px] font-medium transition-colors sm:min-h-0",
                      categoryFilter === c
                        ? "bg-(--color-surface-3) text-(--color-text-primary)"
                        : "text-(--color-text-tertiary) hover:bg-(--color-surface-2)",
                    )}
                  >
                    <span className="h-2 w-2 rounded-full" style={{ background: CATEGORY_SWATCH[c] }} />
                    {CATEGORY_LABELS[c]}
                  </button>
                ))}
              </div>
              <div className="ml-auto flex gap-0.5 rounded-full bg-(--color-surface-2) p-0.5">
                {(["map", "list"] as const).map((v) => (
                  <button
                    key={v}
                    onClick={() => setView(v)}
                    aria-pressed={view === v}
                    className={clsx(
                      "rounded-full px-2.5 py-1 text-[11.5px] font-medium capitalize transition-colors",
                      view === v ? "bg-(--color-surface-0) text-(--color-text-primary)" : "text-(--color-text-tertiary)",
                    )}
                  >
                    {v}
                  </button>
                ))}
              </div>
            </div>

            {graphLoading && <Skeleton className="m-4 h-[420px]" />}
            {graph && view === "map" && (
              <div className="min-h-0 flex-1">
                <ForceGraph
                  nodes={graph.nodes}
                  edges={graph.edges}
                  selectedId={selected?.id ?? null}
                  onSelect={setSelected}
                  categoryFilter={categoryFilter}
                />
              </div>
            )}
            {graph && view === "list" && (
              <ul className="min-h-0 flex-1 overflow-y-auto">
                {visibleAttacks.map((a) => (
                  <li key={a.id}>
                    <button
                      onClick={() => setSelected(a)}
                      className={clsx(
                        "flex w-full items-center gap-3 border-b border-(--color-hairline) px-3 py-2.5 text-left transition-colors",
                        selected?.id === a.id ? "bg-(--color-surface-2)" : "hover:bg-(--color-surface-1)",
                      )}
                    >
                      <span
                        className="h-2 w-2 shrink-0 rounded-full"
                        style={{ background: CATEGORY_SWATCH[a.category] }}
                      />
                      <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-(--color-text-primary)">
                        {a.name}
                      </span>
                      <span className="hidden shrink-0 text-[12px] text-(--color-text-tertiary) sm:block">
                        {CATEGORY_LABELS[a.category]}
                      </span>
                      <StatusBadge status={a.status} />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <IdentifyAgentPanel />
        </div>

        <div className="rounded-(--radius-panel) border border-(--color-hairline) xl:sticky xl:top-6 xl:max-h-[calc(100vh-100px)] xl:overflow-y-auto">
          {selected ? (
            <AttackDetail attack={selected} />
          ) : (
            <p className="p-4 text-[13px] text-(--color-text-tertiary)">Select an attack to inspect it.</p>
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * Coverage as a single stacked bar plus three counts. The three-paragraph
 * explainer this replaces said the same thing at twenty times the length;
 * the definitions now live on hover, for the reader who wants them.
 */
function CoverageBar({ coverage }: { coverage: CoverageSummary }) {
  const { by_status: s, total_attacks: total } = coverage
  const items = [
    { key: "simulated", n: s.simulated, label: "simulated", color: "var(--color-defense)" },
    { key: "modeled", n: s.modeled, label: "modeled", color: "var(--color-tier-review)" },
    { key: "taxonomy_only", n: s.taxonomy_only, label: "mapped only", color: "var(--color-surface-4)" },
  ]
  return (
    <div className="w-full sm:w-[300px]">
      <div className="flex items-baseline justify-between">
        <span className="text-[12px] text-(--color-text-tertiary)">Coverage</span>
        <span className="text-[12px] tabular-nums text-(--color-text-tertiary)">{total} vectors</span>
      </div>
      <div className="mt-1.5 flex h-1.5 overflow-hidden rounded-full">
        {items.map((it) => (
          <div key={it.key} style={{ width: `${(it.n / total) * 100}%`, background: it.color }} />
        ))}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1">
        {items.map((it) => (
          <span
            key={it.key}
            title={STATUS_TITLE[it.key]}
            className="inline-flex cursor-help items-center gap-1.5 text-[12px] text-(--color-text-tertiary)"
          >
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: it.color }} />
            <span className="tabular-nums text-(--color-text-secondary)">{it.n}</span> {it.label}
          </span>
        ))}
      </div>
    </div>
  )
}

function AttackDetail({ attack }: { attack: AttackNode }) {
  return (
    <div>
      <div className="flex items-start justify-between gap-3 border-b border-(--color-hairline) px-4 py-3">
        <div className="min-w-0">
          <h2 className="text-[15px] font-semibold leading-snug text-(--color-text-primary)">{attack.name}</h2>
          <p className="mt-0.5 text-[12px] text-(--color-text-tertiary)">
            {attack.id} · {CATEGORY_LABELS[attack.category] ?? attack.category_name}
          </p>
        </div>
        <StatusBadge status={attack.status} />
      </div>

      <div className="flex flex-col gap-4 px-4 py-4">
        <p className="[overflow-wrap:anywhere] text-[13px] leading-relaxed text-(--color-text-secondary)">
          {attack.mechanism}
        </p>

        <TagRow label="Rails" items={attack.rails} />
        <TagRow label="Channels" items={attack.channels} />
        <BulletRow label="Precursor signals" items={attack.precursor_signals} />
        <BulletRow label="Observable features" items={attack.observable_features} />

        {(attack.simulated_by.length > 0 || attack.detected_by.length > 0) && (
          <div>
            <Label>Implemented by</Label>
            <div className="mt-1.5 flex flex-col gap-0.5">
              {attack.simulated_by.map((s) => (
                <code key={s} className="[overflow-wrap:anywhere] font-mono text-[11.5px] text-(--color-generate)">
                  {s}
                </code>
              ))}
              {attack.detected_by.map((s) => (
                <code key={s} className="[overflow-wrap:anywhere] font-mono text-[11.5px] text-(--color-defense)">
                  {s}
                </code>
              ))}
            </div>
          </div>
        )}

        {attack.atlas_mapping && (
          <div>
            <Label>MITRE ATLAS</Label>
            <div className="mt-1 font-mono text-[11.5px] text-(--color-text-secondary)">{attack.atlas_mapping}</div>
          </div>
        )}

        {attack.grounding.length > 0 && (
          <details className="group">
            <summary className="cursor-pointer list-none text-[11px] font-semibold uppercase tracking-[0.08em] text-(--color-text-tertiary) hover:text-(--color-text-secondary)">
              <span className="inline-block transition-transform group-open:rotate-90">›</span> Grounded in (
              {attack.grounding.length})
            </summary>
            <ul className="mt-1.5 flex flex-col gap-1">
              {attack.grounding.map((g) => (
                <li key={g} className="text-[12px] italic leading-relaxed text-(--color-text-tertiary)">
                  {g}
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </div>
  )
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-(--color-text-tertiary)">{children}</div>
  )
}

function TagRow({ label, items }: { label: string; items: string[] }) {
  if (items.length === 0) return null
  return (
    <div>
      <Label>{label}</Label>
      <div className="mt-1.5 flex flex-wrap gap-1">
        {items.map((item) => (
          <span
            key={item}
            className="rounded-full bg-(--color-surface-2) px-2 py-0.5 text-[11.5px] text-(--color-text-secondary)"
          >
            {item.replace(/_/g, " ")}
          </span>
        ))}
      </div>
    </div>
  )
}

function BulletRow({ label, items }: { label: string; items: string[] }) {
  if (items.length === 0) return null
  return (
    <div>
      <Label>{label}</Label>
      <ul className="mt-1.5 flex flex-col gap-1">
        {items.map((s) => (
          <li key={s} className="flex gap-2 text-[12.5px] leading-relaxed text-(--color-text-secondary)">
            <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-(--color-text-disabled)" />
            {s}
          </li>
        ))}
      </ul>
    </div>
  )
}
