/**
 * App shell, a single top bar over a full-width canvas.
 *
 * The sidebar this replaces carried a pillar name, a screen name and a
 * blurb for every entry: fifteen lines of chrome permanently on screen,
 * and 244px of width taken from the simulation, which is the one thing
 * here worth looking at. A top bar gives the content the full page and
 * states each destination in one word.
 */

import { NavLink, Outlet } from "react-router-dom"
import clsx from "clsx"
import { useApiData } from "../api/hooks"

const NAV = [
  { to: "/live", label: "Live" },
  { to: "/atlas", label: "Attacks" },
  { to: "/studio", label: "Simulation" },
  { to: "/defense", label: "Detection" },
  { to: "/arena", label: "Hardening" },
]

export function AppShell() {
  const health = useApiData<{ status: string }>("/api/health")
  const online = health.data?.status === "ok"

  return (
    <div className="flex h-screen flex-col bg-(--color-canvas)">
      <header className="flex h-[52px] shrink-0 items-center gap-5 border-b border-(--color-hairline) px-4 sm:px-6">
        <div className="flex shrink-0 items-center gap-2.5">
          <JanusMark />
          <span className="text-[15px] font-bold tracking-tight text-(--color-text-primary)">JANUS</span>
        </div>

        {/* The active state is a filled pill rather than an underline
            sitting on the header's bottom rule: `overflow-x-auto` makes
            this a scroll container, which clips in BOTH axes, so an
            underline positioned outside the nav's own box never rendered
            at any width. */}
        <nav className="flex min-w-0 flex-1 items-center gap-0.5 overflow-x-auto" aria-label="Main">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                clsx(
                  "shrink-0 rounded-full px-3 py-1.5 text-[13px] font-medium transition-colors",
                  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--color-focus)",
                  isActive
                    ? "bg-(--color-attack-bg) text-(--color-attack)"
                    : "text-(--color-text-tertiary) hover:bg-(--color-surface-2) hover:text-(--color-text-secondary)",
                )
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="hidden shrink-0 items-center gap-1.5 text-[11.5px] text-(--color-text-tertiary) sm:flex">
          <span
            className={clsx(
              "h-1.5 w-1.5 rounded-full",
              online ? "animate-blink bg-(--color-defense)" : "bg-(--color-surface-4)",
            )}
          />
          {online ? "Engine connected" : "Engine offline"}
        </div>
      </header>

      <main className="min-h-0 min-w-0 flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  )
}

/**
 * The mark: two faces at one doorway.
 *
 * Janus is the two-faced god of doorways, one face toward what is coming
 * and one toward what has passed, which is the structure of this system: a
 * red team looking for the next attack and a blue team looking at every
 * attack already seen. Two arrowheads facing away from each other, and the
 * straight gap between them is the authorization gate both are watching.
 */
function JanusMark() {
  return (
    <svg width="21" height="21" viewBox="0 0 100 100" aria-hidden strokeLinejoin="round" strokeWidth="5">
      <path d="M2 50 L44 8 L44 40 L32 50 L44 60 L44 92 Z" fill="var(--color-brand-red)" stroke="var(--color-brand-red)" />
      <path d="M98 50 L56 8 L56 40 L68 50 L56 60 L56 92 Z" fill="var(--color-brand-teal)" stroke="var(--color-brand-teal)" />
    </svg>
  )
}
