export type AttackStatus = "simulated" | "modeled" | "taxonomy_only"

export interface AttackNode {
  id: string
  kind: "attack"
  category: string
  category_name: string
  name: string
  mechanism: string
  rails: string[]
  channels: string[]
  actors: string[]
  precursor_signals: string[]
  observable_features: string[]
  atlas_mapping: string | null
  status: AttackStatus
  grounding: string[]
  simulated_by: string[]
  detected_by: string[]
}

export interface OtherNode {
  id: string
  kind: "category" | "rail" | "channel" | "signal" | "detector"
  name?: string
}

export type AtlasNode = AttackNode | OtherNode

export interface AtlasEdge {
  source: string
  target: string
}

export interface AtlasGraph {
  nodes: AtlasNode[]
  edges: AtlasEdge[]
}

export interface CategoryCoverage {
  category_name: string
  simulated: number
  modeled: number
  taxonomy_only: number
  total: number
}

export interface CoverageSummary {
  total_attacks: number
  by_status: { simulated: number; modeled: number; taxonomy_only: number }
  by_category: Record<string, CategoryCoverage>
}

export interface EvalReport {
  precision: number
  recall: number
  f1: number
  pr_auc: number
  roc_auc: number
  threshold: number
  n_positive: number
  n_total: number
}

export interface GbmMetrics extends EvalReport {
  train_latency_s: number
  inference_ms_per_1000_rows: number
  dataset: string
  n_rows: number
  n_fraud: number
}

export interface GnnHybridMetrics {
  gnn_only: EvalReport
  hybrid: EvalReport
  device: string
  n_nodes: number
  n_edges: number
  epochs: number
  dataset: string
  n_rows: number
  n_fraud: number
}

export interface MuleRingMetrics extends EvalReport {
  n_injected_rings: number
  caveat: string
}

export interface MuleGeneralization {
  in_distribution_obvious_style: EvalReport
  out_of_distribution_subtle_style: EvalReport
  n_train_rings_obvious: number
  n_test_rings_subtle: number
}

export interface SequenceTransformerMetrics extends EvalReport {
  n_train: number
  n_test: number
}

export interface FidelityFeatureReport {
  feature: string
  wasserstein: number
  ks_statistic: number
  ks_pvalue: number
  js_divergence: number
  real_n: number
  synthetic_n: number
}

export interface FidelityScorecard {
  batch_name: string
  features: FidelityFeatureReport[]
  correlation: { mean_abs_delta: number; max_abs_delta: number; n_features: number } | null
  distinguisher: {
    auc: number
    distance_from_ideal: number
    n_real: number
    n_synthetic: number
    n_shared_rows_excluded?: number
  } | null
  graph_topology: { degree_distribution_ks: number; clustering_coefficient_delta: number } | null
}

export interface TechniqueRoundStats {
  technique: string
  attempts: number
  successes: number
  incompletes: number
  bypass_rate: number | null
  successful_payloads: string[]
}

export interface AgenticRound {
  round_num: number
  firewall_present: boolean
  detector_kind: string | null
  overall_bypass_rate: number | null
  technique_stats: Record<string, TechniqueRoundStats>
}

export interface TabularHardeningRound {
  round: number
  evasion_rate: number
  n_targeted: number
  n_evaded: number
  mean_features_perturbed: number
  mean_l2_perturbation: number
  /** Displacement in per-feature standard deviations, scale-free, unlike
   *  raw L2, which is in whatever units the features happen to carry. */
  mean_perturbation_std_units?: number
  clean_eval: EvalReport
}

export interface CombinedCurvePoint {
  round: number
  arm: "agentic" | "tabular_adversarial"
  attacker_win_rate: number | null
  defense_strength: number | null
  note: string
}

export interface AgenticMeta {
  backend: string | null
  red_team_model: string | null
  shopping_agent_model: string | null
  wall_clock_s: number | null
  total_attempts: number
  template_fallbacks: number
  provenance: string
  llm_call_summary?: Record<string, { calls: number; avg_latency_ms: number; providers: string[] }>
}

export interface ClosedLoopResult {
  generated_at: number
  agentic_rounds: AgenticRound[]
  agentic_meta?: AgenticMeta
  tabular_adversarial: { baseline_eval: EvalReport; rounds: TabularHardeningRound[] }
  combined_curve: CombinedCurvePoint[]
}

export interface LatencyBudget {
  fast_path_ms_budget: number
  fast_path_members: string[]
  async_members: string[]
  note: string
}

// --- live sandbox WebSocket protocol ---

export interface TranscriptEventMsg {
  type: "transcript_event"
  role: string
  content: string
  tool_name: string | null
  tool_args: Record<string, unknown> | null
}

export interface RedTeamPayloadMsg {
  type: "red_team_payload"
  payload: string
  reasoning: string
}

export type Verdict = "pass" | "flag" | "block"

export interface FirewallEventDto {
  stage: string
  verdict: Verdict
  reasons: string[]
}

export interface ResultMsg {
  type: "result"
  attack_succeeded: boolean
  incomplete: boolean
  notes: string[]
  firewall_events: FirewallEventDto[]
}

export interface ErrorMsg {
  type: "error"
  message: string
}

export type SandboxMsg = TranscriptEventMsg | RedTeamPayloadMsg | ResultMsg | ErrorMsg

export interface RunRoundResult {
  technique: string
  firewall_enabled: boolean
  attack_succeeded: boolean
  incomplete: boolean
  notes: string[]
  payload: string
  reasoning: string
}

/* ---------------------------------------------------------------------
 * Stacked ensemble (/api/defend/ensemble)
 * ------------------------------------------------------------------ */

export type DecisionTier = "auto_approve" | "step_up" | "review" | "decline"

/** One decision the stacked model actually made on a held-out row. The
 *  prototype's authorization stream replays these rather than animating
 *  invented numbers, so the tier mix and fraud rate on screen are the
 *  measured ones. */
export interface ScoredDecision {
  amount: number
  hour: number
  product: string
  gbm: number
  gnn: number
  anomaly: number
  risk: number
  tier: DecisionTier
  is_fraud: number
}

export interface TierRow {
  tier: DecisionTier
  n: number
  share_of_volume: number
  n_fraud: number
  share_of_fraud_caught: number
  fraud_rate_within_tier: number
}

export interface TierDistribution {
  n_total: number
  n_fraud: number
  thresholds: { step_up: number; review: number; decline: number } | null
  tiers: TierRow[]
}

export interface EnsembleResult {
  members: Record<string, EvalReport>
  stacked: EvalReport
  member_weights: Record<string, number>
  tier_distribution: TierDistribution
  tier_distribution_capacity?: TierDistribution
  n_train: number
  n_meta: number
  n_test: number
  split_note: string
  wall_clock_s: number
  graph: { n_nodes: number; n_edges: number; device: string; epochs: number }
  scored_sample: ScoredDecision[]
  dataset: string
  n_rows: number
  n_fraud: number
}

export interface LatencyProfile {
  single_row_scoring_ms: { n_samples: number; mean: number; p50: number; p95: number; p99: number; max: number }
  batched_scoring: Record<string, { total_ms: number; ms_per_row: number }>
  n_features: number
  hardware: string
  note: string
}

export interface TimeAblation {
  with_capture_clock: EvalReport & { n_features: number }
  without_capture_clock: EvalReport & { n_features: number }
  note: string
}

export interface OnboardingResult extends EvalReport {
  false_positive_rate_thin_file_legit: number
  false_positive_rate_established_legit: number
  recall_by_ring_sophistication: Record<string, { n: number; recall: number | null }>
  feature_importance: Record<string, number>
  fairness_note: string
  sophistication_note: string
  population: {
    n_applications: number
    n_legit: number
    n_thin_file_legit: number
    n_synthetic: number
    n_rings: number
    synthetic_rate: number
  }
}

export interface ShapExample {
  risk_score: number
  true_label: number
  top_reasons: { feature: string; shap: number }[]
}

export interface Explanations {
  model: string
  base_value: number
  global_mean_abs_shap: { feature: string; mean_abs_shap: number }[]
  examples: ShapExample[]
  note: string
}
