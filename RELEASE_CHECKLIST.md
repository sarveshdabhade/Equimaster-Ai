# Release checklist - 19 November 2026

Recoverable checkpoints: `D:/Project-Backups/Equimaster-Ai-finalise-20261117-145314`
and `D:/Project-Backups/Equimaster-Ai-finalise-20261119-164100`.

| Step | Result | Evidence |
|---|---|---|
| 1. Daily pipeline and scheduler | Passed; duplicate cleanup needs administrator | Registered task result 0; shared lock and Windows atomic-write regression tests |
| 2. Full-universe data | Passed | 59/59 datasets, provider reference 2026-11-19 |
| 3. Model coverage | Passed | 59/59 current bundles, all four horizons, news gate preserved |
| 4. All-stock valuation | Passed | 58/58 fresh scenarios, methods/inputs/assumptions exported |
| 5. Evaluation and inference | Passed | 236 finite inference checks; frozen test and cost reports |
| 6. Browser acceptance | Passed | 10 flows in the lockfile-installed server; desktop/mobile and CSV downloads |
| 7. Documentation | Updated | Current status, research-paper addendum and dated report copies |
| 8. Source release and runtime archive | Completion evidence in the packaged release | RELEASE_MANIFEST.json records the final source commit, successful exact-commit CI, model inventory and file hashes; ZIP CRC and SHA-256 are verified |

Research validation remains separate: future/untouched walk-forward periods,
independent news labels, interval calibration and point-in-time fusion validation.
See PROJECT_STATUS.md for the precise limits and the administrator command.
