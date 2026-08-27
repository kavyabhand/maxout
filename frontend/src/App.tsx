import { HashRouter, Navigate, Route, Routes } from "react-router-dom"
import { AppShell } from "./components/AppShell"
import { LiveScreen } from "./screens/LiveScreen"
import { AttackAtlasScreen } from "./screens/AttackAtlasScreen"
import { SimulationStudioScreen } from "./screens/SimulationStudioScreen"
import { DefenseConsoleScreen } from "./screens/DefenseConsoleScreen"
import { RedBlueArenaScreen } from "./screens/RedBlueArenaScreen"

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Navigate to="/live" replace />} />
          <Route path="/live" element={<LiveScreen />} />
          {/* The landing route moved from /overview to /live; anyone
              holding the old link still lands somewhere real. */}
          <Route path="/overview" element={<Navigate to="/live" replace />} />
          <Route path="/atlas" element={<AttackAtlasScreen />} />
          <Route path="/studio" element={<SimulationStudioScreen />} />
          <Route path="/defense" element={<DefenseConsoleScreen />} />
          <Route path="/arena" element={<RedBlueArenaScreen />} />
          <Route path="*" element={<Navigate to="/live" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}
