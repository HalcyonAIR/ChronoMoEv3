# ChronoMoEv3 Invariants

Constitutional rules for lifecycle edits. These must not be broken.

---

## Irreversibles

**Definition**: Operations that destroy information without exact recovery path.

**Types**:
- Scar append (Phase 3)
- Reflex crystallization (Phase 3)
- Epoch boundary (Phase 3)
- MERGE (Phase 5)

**Rules**:
1. Never allowed in strain or panic bands
2. Must pass `LifecycleGates.require_edit_allowed()` (non-bypassable)
3. Must pass two-step commit (propose → approve → execute)
4. Requires `time_in_comfort > edit_calm_steps` (default 500)

**Enforcement**: `LifecycleGates.require_edit_allowed()` raises `GateViolation` if blocked.

**Location**: `chronomoe_v3/lifecycle_gates.py:LifecycleGates.require_edit_allowed()`

---

## Structural Edits

**Types**: SPAWN, PRUNE, SPLIT, MERGE

**Required for all**:
1. Proposal with `EditEvidence` (predicted ΔF_l, component improvements, trigger)
2. Evidence must clear `MIN_DELTA_F = 0.01` or `MIN_DELTA_PSI = 0.02`
3. Dry-run prediction (if available)
4. Two-step commit: proposed at step N, executed at N+1 if improvement holds
5. Audit logging: PROPOSED, APPROVED (or REJECTED), EXECUTED

**"Do nothing" is valid**:
- If predicted ΔF_l < MIN_DELTA_F and ΔPsi < MIN_DELTA_PSI, reject with reason `insufficient_benefit`
- This rejection must be logged in audit trail

**Enforcement**:
- Evidence thresholds: `DryRunEvaluator.__init__(min_f_l_improvement=MIN_DELTA_F, min_psi_improvement=MIN_DELTA_PSI)`
- Rejection reason: `DryRunEvaluator.evaluate_spawn()` returns `improvement_confirmed=False, reason="insufficient_benefit"`
- Gate check: `EditExecutor.propose_*()` calls `gates.require_edit_allowed()`

**Location**:
- Evidence: `chronomoe_v3/edit_proposals.py:EditEvidence`
- Thresholds: `chronomoe_v3/dry_run_evaluator.py:MIN_DELTA_F, MIN_DELTA_PSI`
- Executor: `chronomoe_v3/edit_executor.py:EditExecutor`

---

## Mask Consistency

**Single-source definition**:
```python
active_mask = utilization >= min_tokens
```

**Rules**:
1. Computed once per free energy call
2. Any metric using a different mask must name it and justify
3. Redundancy uses stricter threshold: `redundancy_mask = utilization >= redundancy_min_tokens`

**Rationale**: Prevents double-masking bugs where metrics accidentally exclude different expert sets.

**Enforcement**: Code review + documentation in `compute_free_energy()`.

**Location**: `chronomoe_v3/free_energy.py:compute_free_energy()`

**Planned**: Add assertion that `active_mask.sum() == num_active_experts` before return.

---

## Non-Bypassability

**Rules**:
1. Lifecycle gates must be checked at executor boundary
2. No other code path may perform structural edits
3. EditExecutor is the only entry point for SPAWN/PRUNE/SPLIT/MERGE
4. All edit methods must call `gates.require_edit_allowed()` before creating proposal

**Enforcement**:
- `EditExecutor.propose_spawn()`: calls `gates.require_edit_allowed()` at line 1
- `EditExecutor.propose_prune()`: calls `gates.require_edit_allowed()` at line 1
- `EditExecutor.propose_split()`: calls `gates.require_edit_allowed()` at line 1
- `EditExecutor.propose_merge()`: calls `gates.require_edit_allowed()` at line 1
- `gates.require_edit_allowed()`: raises `GateViolation` exception if blocked (cannot be silently ignored)

**Violation**: If edit appears in model without passing through EditExecutor, this is an architectural violation.

**Location**: `chronomoe_v3/edit_executor.py:EditExecutor`

---

## Forensics

**Every proposal log must include**:
- `step`: Training step at proposal
- `stress_band`: Current band (comfort/strain/panic)
- `time_in_comfort`: Steps in comfort band
- `gates_state`: `{allow_scar, allow_crystallize, allow_edit}`
- `evidence`: Full `EditEvidence.to_dict()` (predicted deltas, trigger)
- `f_l`: Current free energy

**Every execution log must include**:
- All proposal fields (above)
- `actual_delta_f_l`: Actual improvement (if measured)
- `guardrail_warning`: Any pre-execution warnings (e.g., utilization ratio for MERGE)

**Rejection logs must include**:
- `reason`: Why rejected (`gates_changed`, `no_improvement`, `insufficient_benefit`, `final_gate_check_failed`)
- `dry_run`: Full DryRunResult if applicable

**Enforcement**: `EditExecutor._log_event()` creates `AuditLogEntry` with required fields.

**Location**: `chronomoe_v3/edit_proposals.py:AuditLogEntry`, `chronomoe_v3/edit_executor.py:EditExecutor._log_event()`

**Verification**: Audit log must be valid JSONL, one event per line, parseable for post-mortems.

---

## Guardrails

**MERGE utilization ratio**:
- If `max(util_a, util_b) / min(util_a, util_b) > 10.0`, log WARNING
- Warning recorded in audit trail: `guardrail_warning.guardrail = "utilization_ratio_warning"`
- Does not block execution (forensic power, not enforcement)

**Location**: `chronomoe_v3/edit_executor.py:EditExecutor.execute_merge()`

**Planned**:
- PRUNE starvation check: reject if removing expert causes `Neff < neff_critical`
- SPAWN capacity check: reject if layer already at `max_experts`
- SPLIT bimodality threshold: require `bimodality > 0.7` to justify split

---

## Reversibility Spectrum

**Strong reversible** (exact state restoration):
- SPLIT: Can merge back to recover original expert

**Weak reversible** (capacity reintroduction, history lost):
- SPAWN: Can prune if doesn't help (additive, not reversible)
- PRUNE: Can re-spawn capacity, but original expert's learned state is gone

**Irreversible** (destructive):
- MERGE: Averaged parameters cannot be unmixed, personality lost permanently

**Enforcement**: Documentation only (semantic, not runtime).

**Location**: `PROGRESS.md:Reversibility Spectrum`

---

## Spawn Strategy

**Current**: `clone_seeded` (clone parent + perturbation)

**Planned**: `blank_spawn_with_probation` (random init, learn from scratch)

**Rule**: Strategy must be explicit in `SpawnProposal.spawn_strategy` field.

**Enforcement**: `SpawnProposal.spawn_strategy` property returns `"clone_seeded"` (default) or strategy from details.

**Location**: `chronomoe_v3/edit_proposals.py:SpawnProposal.spawn_strategy`

---

## Violations

If any invariant is violated:
1. Search audit log for edit that bypassed gates
2. Check if `EditExecutor` was entry point
3. Verify `gates.require_edit_allowed()` was called
4. Check if gate state changed between proposal and execution
5. If violation is architectural (edit without executor), file bug with severity: CRITICAL

**Remedy**: All structural changes must route through `EditExecutor`. No exceptions.
