# Teacher–student cycle design

These are unimplemented extension contracts. Core inference and already deployed models remain independent. Implement after [M1 baselines](ga-measurements.md) and the capture/ownership repairs. Only shared native runtime-adapter work additionally requires [N1 controls](ga-measurements.md). The [specialist portfolio](specialist-portfolio.md) is a separate extension requiring measured demand.

### 9.0 Independent opt-in capabilities

- **Change / owners:** serving lock, Capture, cycle request and deployment own model bindings, observation, enabled job stages and promotion respectively. Dynamic portfolio routing remains in the separate design. Plain inference stays default; absent capabilities create no listeners, logs, queues, model loads, optional imports or recurring work. Loading pretrained deployments needs no trainer; offline fitting needs no live capture/teacher. Document configurations as examples, not alternate engines or mode enums.
- **Verify:** core-only inference, capture-only, monitor-only, offline fit and candidate-only publication. Missing selected inputs fail at entry; disabled capabilities do zero work and training/monitor failures leave the incumbent unchanged.

### 9.1 Bounded cycle

- **Change / owner:** semantic-learn adds one `cycle --request FILE --to DIRECTORY` that runs once and exits; an external timer controls recurrence. Request binds frozen task, incumbent, observations, enabled stages, recipe/declaration and record/byte/teacher-call/fit limits. Freeze a complete-line capture prefix with source identity, offsets, event IDs and content digest through files publication. Existing Journal stores stage references/counts; collect/review/round/pipeline remain stage owners.
- **Verify:** resume at materialization/teacher/fit/compare boundaries without repeated committed work; concurrent append preserves the frozen prefix, replacement refuses. Advance input cursor only with committed materialization; candidate publication never implicitly promotes.

### 9.2 Sampling populations

- **Change / owner:** capture/selection retains separate random and targeted strata with unit, window/seed, reason, probability and family links. Existing situation sampling is cluster sampling: event multiplicity remains in reports, training claims merge once. Define the finite eligible window before sampling/teacher quotas; retain unresolved/drop denominators. Unknown probabilities or incomplete samples cannot support a full-window rate estimate.
- **Verify:** replay membership, duplicate events, targeted enrichment, budget censoring, overlapping strata and family-connected train/check assignments. Random and targeted reports remain distinguishable.

### 9.3 Teacher evidence

- **Change / owner:** collect scores the same frozen task/redacted inputs blind to student answers, with teacher identity pinned. Accepted answers become judged training candidates; uncertain/failed answers enter review. Keep permissions and reference ancestry; verified corrections use the existing successor/resolution operation. Start with the current hard-label objective.
- **Verify:** confident wrong teacher remains judged, model drift cannot mix a cycle's labels, restricted ancestry fails admission, and redaction/context mismatches are disclosed rather than called student regression.

### 9.4 Monitoring

- **Change / owner:** metrics aggregates stored paired rows: both-accepted agreement/disagreement, each side's acceptance/withholding, failures, verified correctness, cost and latency. Report random/targeted strata separately; mass/margins are task/execution-scoped telemetry. Predeclare fixed windows, baseline uncertainty, tolerance, minimum evidence and repeated-look policy; retain sparse/degenerate/inconclusive outcomes. Sequential inference is deferred until measured teacher-call costs justify a time-uniform method.
- **Verify:** increased withholding cannot hide behind accepted precision; high-mass wrong answers never become correctness evidence. Counts/weights reconcile, sparse/correlated windows remain inconclusive and no-change simulations check the declared alert horizon.

### 9.5 Development and final evaluation

- **Change / owner:** existing assignments and EvidenceContext separate development optimization from final frozen checks. Freeze arms/acceptance/scope before final consultation; selector consumes development results, not final per-example labels. A final result used to change a candidate requires fresh independent evidence or an explicitly reused comparison. Replay reads existing rows without new inference/exposure.
- **Verify:** adaptive reuse cannot claim fresh independence; missing history remains unknown; family bridges and corrections update assignment. Ledger disclosure is preserved without pretending bookkeeping prevents overfitting.

### 9.6 Bounded semantic triage

- **Change / owner:** an optional frozen semantic switch classifies review priority/failure category. Cycle chooses one bounded recipe at a time using development results; exact code owns budgets, splits and acceptance. Preserve criterion meaning for ordinary LoRA/data experiments; authorized meaning changes reassess affected labels. Record triage/task/execution and recipe/evidence identities.
- **Verify:** uncertain triage produces review, not labels; compute exhaustion starts no further fit. Criterion-changing guidance cannot reuse old current labels; no candidate selection reads final-check labels.

### 9.7 Promotion and continued observation

- **Change / owner:** deployment's existing comparison/decision/pointer transaction binds candidate, expected incumbent and declared quality/coverage/resource constraints. Extend that reducer for missing required measurements; preserve inconclusive results. Candidate-only is default; automatic promotion requires explicit policy. Retain prior entry for rollback and start monitoring under the new execution identity.
- **Verify:** stale/unsupported decisions leave serving unchanged; resume cannot promote twice. Promotion/rollback use one owner. Automatic operation requires M1's actual serving/fit baseline; shared native reuse additionally needs N1's passing scope.

