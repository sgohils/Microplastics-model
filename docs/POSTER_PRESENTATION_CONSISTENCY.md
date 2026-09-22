# Poster-Presentation Consistency Checklist

Ensure that the poster and presentation use the same core scientific content to avoid contradictions.

| Item | Poster Check | Presentation Check | Notes |
|------|--------------|---------------------|-------|
| Title | Matches exactly | Matches exactly | Should be identical; if shortened for poster, ensure it conveys the same meaning. |
| Research Question | Matches exactly | Matches exactly | Must be verbatim or minimally altered for space; meaning unchanged. |
| Hypothesis | Matches exactly | Matches exactly | Primary and null hypothesis should be consistent. |
| Study Region | Same description | Same description | Spatial extent (Delaware River Basin, validation regions) and temporal focus (July 2018–March 2019). |
| Dataset | Same sources and observation count | Same sources and observation count | USGS microplastic datasets, ~150-200 observations. |
| Target Variable | Same definition and units | Same definition and units | Microplastic concentration in river water (particles/m³), log1p transformed for training. |
| Model Architecture | Same name and key details | Same name and key details | Spatiotemporal GraphSAGE-GRU (`graphsage_gru`), feature encoder → GRU → GraphSAGE → prediction head. |
| Baselines | Same list | Same list | Mean/Median, Ridge, Random Forest, XGBoost, MLP. |
| Experimental Design | Same splits described | Same splits described | Random split, spatial holdout, temporal holdout, spatiotemporal holdout, ablation, sparse monitoring, uncertainty quantification. |
| Primary Metrics | Same metrics reported | Same metrics reported | MAE, RMSE, R², uncertainty (95% PI coverage), statistical significance (p-value). |
| Key Results | Same numbers and interpretation | Same numbers and interpretation | Use exact values from `experiments/results/` (e.g., final_summary.json, statistical_comparison.json). |
| Limitations | Same list | Same list | Data scarcity, cross-sectional sampling, static graph, uncertainty under-dispersion, geographic focus. |
| Conclusions | Same take-home messages | Same take-home messages | GNN did not significantly outperform XGBoost; modest contributions from graph and temporal; spatial > temporal generalization. |
| Figures | Same source data | Same source data | Any plot or map must be generated from the same underlying data and code. |
| Numerical Values | Every number matches source | Every number matches source | Audit all numbers against `experiments/results/`, `MODEL_CARD.md`, and other authoritative sources. |
| Terminology | Consistent use of terms | Consistent use of terms | E.g., "microplastic concentration", "river network", "graph neural network", "spatiotemporal". |
| Citations & References | Same sources cited | Same sources cited | If references are included, they must match (e.g., USGS datasets, key literature). |
| AI Usage Disclosure | Same statement | Same statement | If required by competition, the disclosure must match what is in `docs/AI_USAGE_LOG.md`. |

## Instructions
1. Complete your poster and presentation drafts.
2. Go through each item in this checklist and verify consistency.
3. For any mismatches, decide which version is correct (refer to source documents) and update both poster and presentation accordingly.
4. Pay special attention to numerical values, figure captions, and axis labels.
5. When in doubt, consult the following source documents in order of priority:
   - `experiments/results/` (JSON files)
   - `MODEL_CARD.md`
   - `docs/IMPLEMENTATION_SPEC.md`
   - `docs/ASSUMPTIONS.md`
   - `docs/RESEARCH_QUESTION_EXPLANATION.md`
   - `docs/HYPOTHESIS_EXPLANATION.md`
   - `docs/METHODS.md`

## Final Sign-off
After completing the checklist, the student should be able to confirm:
> "All scientific content in my poster and presentation is consistent with the actual project files and experiment results, with no fabricated or exaggerated claims."