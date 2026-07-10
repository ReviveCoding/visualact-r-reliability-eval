# Failure Taxonomy

| Family | Observable symptom | Detection | Fallback / response |
|---|---|---|---|
| Source mismatch | image-label join < 1.0 | archive preflight | HOLD |
| Duplicate leakage | pHash cluster appears in multiple splits | split integrity report | rebuild split |
| pHash chaining | A~B and B~C merge scenes with A not similar to C | complete-linkage cluster test | complete-link pHash clustering, not transitive union-find |
| Degenerate final set | frozen final lacks hard/clean or action diversity | split composition gate | rebuild split before any tuning |
| Raw detector unavailable | only post-NMS outputs are exposed | detector cache validation | do not run CEF; HOLD |
| CEF overreach | coverage improves while contamination rises | dev coverage-overreach gate | use C3 NMS baseline |
| Redundant evidence | Top-K crops are near duplicates | crop redundancy metric / MMR test | CEF diversity penalty |
| Schema failure | invalid tool JSON or invalid crop ID | strict parser / rollout trace | typed REVIEW fallback |
| Missing trace | a final prediction is absent | trace coverage metric | REVIEW fallback and HOLD release |
| RCTR future leakage | terminal fields, labels, or outcomes enter prefix features | unit test, routing report | HOLD |
| Unmatched routing | C6 uses a different review rate from confidence guard | matched-capacity gate | HOLD C6 claim |
| Weak paired evidence | final matched sample < configured minimum | paired comparison artifact | `INSUFFICIENT_SAMPLE`, no uplift claim |
| GPU incompatibility | model load, OOM, or adapter attachment fails | local preflight/rollout manifest | record hold; use smaller model or reduce visual token budget |
| Unexecuted counterfactual | no repeated crop-removal rollout exists | counterfactual artifact | prohibit causal-grounding wording |
