# Data Card

**Primary external source:** User-supplied BDD100K validation image/label assets. The project treats this as a BDD100K-derived internal research pool and does not use the official BDD leaderboard split as its claim protocol.

**Observation unit:** One image × triage task template.

**At inference:** Image, detector candidates, task prompt, generated crops.

**Offline only:** GT boxes/classes, occlusion/truncation, deterministic policy label, supporting object IDs, weather/scene/time metadata unless explicitly configured as descriptive slices.

**Repository policy:** No BDD images, labels, or crops may be committed. External assets must be acquired and extracted by the user. Source preflight manifests may be retained because they contain hashes/counts, not media.

**Known limitations:** BDD labels are perception annotations, not human scene-review decisions. Triage actions are deterministic geometry-policy proxies and not driving actions.
