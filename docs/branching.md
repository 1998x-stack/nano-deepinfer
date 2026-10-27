# Branch / merge / release strategy

1. `main` is always the latest successfully verified revision, not a future roadmap branch. No forced updates.
2. Milestone pointers `v0.1`, `v0.2`, `v0.3`, `v1.0` are established from the first Stage 00 merge. They initially point to the **same commit**, and are not release indicators.
3. Use short-lived `feat/stage-00-foundation`, `feat/stage-01-dense-reference` etc. with PR into `main`; squash merging allowed.
4. When a milestone stream advances, sync it with current `main` using fast-forward where safe. Do not merge a future branch into main just to keep markers moving.
5. Immutable version tags (e.g. `v0.1.0`) are created only when corresponding acceptance gates pass; never confuse branch names with published versions.
6. PR must contain changed APIs, tests, baseline/evidence, limitations, dependency/attribution review and next gate.
7. Add branch protection/rulesets separately if owner permissions allow; repository admin may need to configure required CI.

