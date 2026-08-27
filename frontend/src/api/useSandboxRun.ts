/**
 * Runs one agentic attack against the sandbox and returns its transcript.
 *
 * This replaces a WebSocket client. The socket streamed events as the
 * session produced them, but the diagram consuming them stages its own
 * reveal at a fixed cadence regardless; it never rendered at socket
 * speed, so a single request that returns the whole transcript looks
 * identical on screen and removes the only long-lived connection in the
 * app. That is what lets the prototype deploy as a static build with
 * recorded transcripts and behave the same way.
 *
 * `source` distinguishes a live run from a recorded one so the UI can say
 * which it is showing.
 */

import { useCallback, useRef, useState } from "react"
import { postJson, recordedSandboxRun } from "./client"
import type { SandboxMsg } from "./types"

export type SandboxStatus = "idle" | "running" | "done" | "error"
export type SandboxSource = "live" | "recorded" | null

interface RunResponse {
  events?: SandboxMsg[]
  agent?: string
  error?: string
}

export function useSandboxRun() {
  const [status, setStatus] = useState<SandboxStatus>("idle")
  const [events, setEvents] = useState<SandboxMsg[]>([])
  const [source, setSource] = useState<SandboxSource>(null)
  const [agent, setAgent] = useState<string | null>(null)

  // Guards a slow response from a superseded run landing after a newer
  // one has already started, without it, re-clicking "Run attack"
  // could let the previous request overwrite the current transcript.
  const runId = useRef(0)

  const run = useCallback(async (technique: string, firewallEnabled: boolean) => {
    const id = (runId.current += 1)
    setEvents([])
    setSource(null)
    setAgent(null)
    setStatus("running")

    const res = await postJson<RunResponse>("/api/sandbox/run", {
      technique,
      firewall_enabled: firewallEnabled,
    })
    if (id !== runId.current) return

    if (res?.events?.length) {
      setEvents(res.events)
      setAgent(res.agent ?? null)
      setSource("live")
      setStatus("done")
      return
    }

    const recorded = recordedSandboxRun(technique, firewallEnabled) as SandboxMsg[] | null
    if (recorded?.length) {
      setEvents(recorded)
      setAgent("scripted policy agent")
      setSource("recorded")
      setStatus("done")
      return
    }
    setStatus("error")
  }, [])

  const reset = useCallback(() => {
    runId.current += 1
    setEvents([])
    setSource(null)
    setAgent(null)
    setStatus("idle")
  }, [])

  return { status, events, source, agent, run, reset }
}
