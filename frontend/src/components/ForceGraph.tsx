import { useEffect, useRef } from "react"
import * as d3 from "d3"
import type { AtlasEdge, AtlasNode, AttackNode } from "../api/types"

interface ForceGraphProps {
  nodes: AtlasNode[]
  edges: AtlasEdge[]
  selectedId: string | null
  onSelect: (node: AttackNode) => void
  categoryFilter: string | null
}

type SimNode = d3.SimulationNodeDatum & AtlasNode
type SimLink = d3.SimulationLinkDatum<SimNode>

const CATEGORY_COLOR: Record<string, string> = {
  A: "#1f4b8e",
  B: "#f79e1b",
  C: "#ff5f00",
  D: "#eb001b",
}

const KIND_RADIUS: Record<string, number> = {
  attack: 9,
  category: 15,
  rail: 5,
  channel: 5,
  signal: 4,
  detector: 6,
}

function nodeColor(n: AtlasNode): string {
  if (n.kind === "attack") return CATEGORY_COLOR[(n as AttackNode).category] ?? "#98a0ae"
  if (n.kind === "category") return "var(--color-text-secondary)"
  if (n.kind === "detector") return "var(--color-defense)"
  return "var(--color-border-strong)"
}

export function ForceGraph({ nodes, edges, selectedId, onSelect, categoryFilter }: ForceGraphProps) {
  const svgRef = useRef<SVGSVGElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return
    const width = containerRef.current.clientWidth
    const height = containerRef.current.clientHeight || 560

    const visibleNodes: SimNode[] = (nodes as SimNode[]).filter((n) => {
      if (!categoryFilter) return true
      if (n.kind === "attack") return (n as unknown as AttackNode).category === categoryFilter
      if (n.kind === "category") return n.id === `category:${categoryFilter}`
      return true
    })
    const visibleIds = new Set(visibleNodes.map((n) => n.id))
    // d3-force's forceLink mutates link objects in place, replacing string
    // source/target with resolved node object references; filtering and
    // then shallow-copying here (rather than filtering `edges` directly)
    // stops that mutation from corrupting the `edges` prop across renders,
    // which would otherwise break this same string-based filter the next
    // time the effect re-runs (e.g. toggling the category filter).
    const edgeId = (v: string | SimNode): string => (typeof v === "string" ? v : v.id)
    const visibleEdges = edges
      .filter((e) => visibleIds.has(edgeId(e.source as unknown as string | SimNode)) && visibleIds.has(edgeId(e.target as unknown as string | SimNode)))
      .map((e) => ({ source: edgeId(e.source as unknown as string | SimNode), target: edgeId(e.target as unknown as string | SimNode) }))

    const svg = d3.select(svgRef.current)
    svg.selectAll("*").remove()
    svg.attr("viewBox", `0 0 ${width} ${height}`)

    const zoomLayer = svg.append("g")

    // Auto-fitting stops the moment the viewer takes control. `sourceEvent`
    // is set only for a real gesture, so the programmatic transforms below
    // do not switch it off themselves.
    let autoFit = true
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 3])
      .on("zoom", (event) => {
        if (event.sourceEvent) autoFit = false
        zoomLayer.attr("transform", event.transform)
      })
    svg.call(zoom)

    // Seed starting positions near the center instead of D3's default
    // (a wide pseudo-random spiral from the origin), with 76 nodes and a
    // strong repulsion force, starting far apart made the layout settle
    // into a stable state that extended well beyond the container's
    // bounds, off the top of the viewBox.
    for (const n of visibleNodes) {
      if (n.x === undefined) n.x = width / 2 + (Math.random() - 0.5) * 40
      if (n.y === undefined) n.y = height / 2 + (Math.random() - 0.5) * 40
    }

    const simulation = d3
      .forceSimulation(visibleNodes)
      .force(
        "link",
        d3
          .forceLink<SimNode, SimLink>(visibleEdges as unknown as SimLink[])
          .id((d) => (d as SimNode).id)
          .distance((l) => {
            const target = l.target as unknown as SimNode
            return target.kind === "category" ? 90 : 55
          })
          .strength(0.5),
      )
      .force("charge", d3.forceManyBody().strength(-160))
      .force("center", d3.forceCenter(width / 2, height / 2))
      // forceCenter only recenters the CENTROID of all nodes, so it does
      // nothing about the graph's disconnected components drifting apart
      // from each other: nothing links them, so only these per-node pulls
      // hold them together. They used to be very weak (0.04) because they
      // were the only thing keeping the layout inside the viewBox and
      // anything stronger crushed its structure. Now that the view is
      // fitted to the layout every tick, they no longer have that job, and
      // can be set to what actually reads well: firm enough that the four
      // category clusters sit near each other, so the fit does not have to
      // zoom out past legibility to frame two islands with a void between.
      .force("x", d3.forceX(width / 2).strength(0.11))
      .force("y", d3.forceY(height / 2).strength(0.13))
      .force(
        "collide",
        d3.forceCollide<SimNode>((d) => KIND_RADIUS[d.kind] + 6),
      )

    const link = zoomLayer
      .append("g")
      .selectAll("line")
      .data(visibleEdges)
      .join("line")
      .attr("stroke", "var(--color-border-strong)")
      .attr("stroke-width", 1)
      .attr("opacity", 0.35)

    const node = zoomLayer
      .append("g")
      .selectAll<SVGCircleElement, SimNode>("circle")
      .data(visibleNodes)
      .join("circle")
      .attr("r", (d) => KIND_RADIUS[d.kind])
      .attr("fill", (d) => nodeColor(d))
      .attr("stroke", (d) => (d.id === selectedId ? "var(--color-text-primary)" : "var(--color-surface-0)"))
      .attr("stroke-width", (d) => (d.id === selectedId ? 2 : 1.5))
      .style("cursor", (d) => (d.kind === "attack" ? "pointer" : "default"))
      .attr("tabindex", (d) => (d.kind === "attack" ? 0 : null))
      .attr("role", (d) => (d.kind === "attack" ? "button" : null))
      .attr("aria-label", (d) => (d.kind === "attack" ? `${d.id}: ${(d as unknown as AttackNode).name}` : null))
      .on("click", (_event, d) => {
        if (d.kind === "attack") onSelect(d as unknown as AttackNode)
      })
      .on("keydown", (event: KeyboardEvent, d) => {
        if (d.kind === "attack" && (event.key === "Enter" || event.key === " ")) {
          event.preventDefault()
          onSelect(d as unknown as AttackNode)
        }
      })
      .on("focus", function () {
        d3.select(this).attr("stroke", "var(--color-focus)").attr("stroke-width", 2.5)
      })
      .on("blur", function (_event, d) {
        d3.select(this)
          .attr("stroke", d.id === selectedId ? "var(--color-text-primary)" : "var(--color-surface-0)")
          .attr("stroke-width", d.id === selectedId ? 2 : 1.5)
      })
      .call(
        d3
          .drag<SVGCircleElement, SimNode>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.2).restart()
            d.fx = d.x
            d.fy = d.y
          })
          .on("drag", (event, d) => {
            d.fx = event.x
            d.fy = event.y
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0)
            d.fx = null
            d.fy = null
          }),
      )

    node.append("title").text((d) => (d.kind === "attack" ? `${d.id}, ${(d as unknown as AttackNode).name}` : d.id.split(":")[1] ?? d.id))

    const labels = zoomLayer
      .append("g")
      .selectAll("text")
      .data(visibleNodes.filter((n) => n.kind === "attack"))
      .join("text")
      .text((d) => d.id)
      .attr("font-size", 9)
      .attr("font-family", "var(--font-mono)")
      .attr("fill", "var(--color-text-tertiary)")
      .attr("dy", -13)
      .attr("text-anchor", "middle")
      .style("pointer-events", "none")

    /**
     * Keep the layout in frame, on every tick rather than once at the end.
     *
     * Centering forces alone are not enough: forceCenter only moves the
     * centroid, and the per-node forceX/forceY pulls backing it up have to
     * stay weak or they crush the structure into a blob. So the settled
     * bounding box is measured and one zoom transform is applied to fit it.
     *
     * Doing that only on `simulation.on("end")` was the bug: with 77 nodes
     * the simulation needs roughly five seconds to cool below alphaMin, and
     * for those five seconds the graph rendered at whatever scale and offset
     * the raw simulation coordinates happened to have, which put a large
     * part of it outside the frame. Fitting every tick is O(n) on 77 nodes,
     * costs nothing, and means the graph is inside its panel from the first
     * frame drawn.
     */
    const fitToFrame = (): d3.ZoomTransform | null => {
      let minX = Infinity
      let maxX = -Infinity
      let minY = Infinity
      let maxY = -Infinity
      for (const n of visibleNodes) {
        if (!Number.isFinite(n.x!) || !Number.isFinite(n.y!)) continue
        const r = KIND_RADIUS[n.kind]
        if (n.x! - r < minX) minX = n.x! - r
        if (n.x! + r > maxX) maxX = n.x! + r
        if (n.y! - r < minY) minY = n.y! - r
        if (n.y! + r > maxY) maxY = n.y! + r
      }
      if (!Number.isFinite(minX) || !Number.isFinite(minY)) return null

      // Padding leaves room for the id label that sits above each attack
      // node, which is drawn outside the circle's own radius.
      const pad = 22
      const boxW = maxX - minX + pad * 2
      const boxH = maxY - minY + pad * 2
      if (boxW <= 0 || boxH <= 0) return null

      const scale = Math.min(width / boxW, height / boxH, 1.6)
      const tx = width / 2 - scale * (minX - pad + boxW / 2)
      const ty = height / 2 - scale * (minY - pad + boxH / 2)
      return d3.zoomIdentity.translate(tx, ty).scale(scale)
    }

    simulation.on("end", () => {
      if (!autoFit) return
      const transform = fitToFrame()
      if (!transform) return
      // Routed through the zoom behaviour on the final fit so a later pan
      // or zoom continues from the fitted view instead of snapping back to
      // the identity transform on first drag.
      svg.transition().duration(300).call(zoom.transform, transform)
    })

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as unknown as SimNode).x!)
        .attr("y1", (d) => (d.source as unknown as SimNode).y!)
        .attr("x2", (d) => (d.target as unknown as SimNode).x!)
        .attr("y2", (d) => (d.target as unknown as SimNode).y!)
      node.attr("cx", (d) => d.x!).attr("cy", (d) => d.y!)
      labels.attr("x", (d) => d.x!).attr("y", (d) => d.y!)

      if (autoFit) {
        const transform = fitToFrame()
        // Set on the layer directly rather than through zoom.transform:
        // the behaviour's own transition/event machinery is not worth
        // running sixty times a second, and the final fit re-syncs it.
        if (transform) zoomLayer.attr("transform", transform.toString())
      }
    })

    return () => {
      simulation.stop()
    }
  }, [nodes, edges, selectedId, onSelect, categoryFilter])

  return (
    <div ref={containerRef} className="h-full w-full">
      <svg ref={svgRef} className="h-full w-full" role="img" aria-label="Attack Atlas force-directed graph" />
    </div>
  )
}
