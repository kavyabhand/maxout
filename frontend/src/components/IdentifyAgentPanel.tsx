import { useState } from "react"
import { postJson } from "../api/client"

interface ProposalResult {
  topic_query: string
  backend: string
  grounding: { source_id: string; score: number }[]
  draft: string
  status: string
}

/**
 * The Identify agent: retrieval and clustering over a local grounding
 * corpus proposes a candidate attack node for human review. It stays on
 * the Attacks screen because it is what keeps the taxonomy a running
 * pipeline rather than a static list.
 */
export function IdentifyAgentPanel() {
  const [query, setQuery] = useState("agent-to-agent negotiation trust")
  const [result, setResult] = useState<ProposalResult | null>(null)
  const [loading, setLoading] = useState(false)

  async function propose() {
    setLoading(true)
    const res = await postJson<ProposalResult>("/api/identify/propose", { topic_query: query })
    setResult(res)
    setLoading(false)
  }

  return (
    <section className="shrink-0 rounded-(--radius-panel) border border-(--color-hairline)">
      <div className="flex flex-wrap items-center gap-2 px-3 py-2.5">
        <span className="text-[12.5px] font-semibold text-(--color-text-primary)">Propose a new vector</span>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="min-w-0 flex-1 rounded-(--radius-control) border border-(--color-hairline) bg-(--color-surface-1) px-3 py-1.5 text-[12.5px] text-(--color-text-primary) outline-none transition-colors focus:border-(--color-focus)"
          placeholder="topic to investigate…"
        />
        <button
          onClick={propose}
          disabled={loading}
          className="shrink-0 rounded-full bg-(--color-text-primary) px-3.5 py-1.5 text-[12.5px] font-semibold text-white transition-colors hover:bg-black disabled:cursor-not-allowed disabled:bg-(--color-text-disabled) focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--color-focus)"
        >
          {loading ? "Proposing…" : "Propose"}
        </button>
      </div>

      {result && (
        <div className="border-t border-(--color-hairline) px-3 py-2.5">
          <div className="flex flex-wrap items-center gap-2 text-[11px]">
            <span className="text-(--color-text-tertiary)">backend {result.backend}</span>
            <span className="font-semibold text-(--color-tier-review)">pending human review</span>
          </div>
          <p className="mt-1.5 text-[12.5px] leading-relaxed text-(--color-text-secondary)">{result.draft}</p>
          <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1">
            {result.grounding.map((g) => (
              <span key={g.source_id} className="font-mono text-[11px] text-(--color-text-tertiary)">
                {g.source_id} · {g.score.toFixed(2)}
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}
