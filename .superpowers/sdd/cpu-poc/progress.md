# SDD ledger — plan: docs/superpowers/plans/2026-11-01-cpu-poc-plan.md
Pre-flight: T2 types/geometry consumed by T3/T6/T7 — signatures as in plan, no conflicts. T4 registry manifests consumed by T7 runner. T5 encoder/gallery consumed by T7. FaceHit defined in types.py (T2) though plan lists it under T5 interface — Ruling: define dataclass in types.py, encoder in face.py imports it; same fields.
Task 1: complete - cli tests green after editable install fix (empty first wheel)
Task 2: complete
Task 3: complete
Task 4: complete
Task 4: Ruling: registry-conformance (manifest->node class) test deferred to Task 7 when runner factories exist — manifests only in Task 4 — cost if wrong: a manifest without implementation would pass validation until Task 7 wiring lands.
Task 5: complete (Labkom frame: 2 usable faces measured)
