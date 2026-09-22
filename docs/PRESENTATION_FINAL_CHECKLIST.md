# Presentation Final Checklist

Run through this checklist before delivering your presentation to ensure scientific rigor, clarity, and readiness.

## Scientific Rigor
[ ] Research question is verbatim from `docs/RESEARCH_QUESTION_EXPLANATION.md` or `docs/PRESENTATION_SPECIFICATION.md`.
[ ] Hypothesis (primary and null) matches `docs/HYPOTHESIS_EXPLANATION.md`.
[ ] Study region description matches `docs/microplastic_gnn_research_specification.md`, Section 9 (STUDY REGION).
[ ] Dataset sources and observation counts match `docs/microplastic_gnn_research_specification.md`, Section 7 (DATA FEASIBILITY AUDIT).
[ ] Target variable definition and units match `docs/microplastic_gnn_research_specification.md`, Section 8 (TARGET VARIABLE).
[ ] Model architecture name and key components match `docs/microplastic_gnn_research_specification.md`, Section 16 (ARCHITECTURE) and `docs/MODEL_CARD.md`.
[ ] Baseline models list matches `docs/IMPLEMENTATION_SPEC.md`, Section 17 (BASELINE MODELS) and `docs/METHODS.md`, Baselines.
[ ] Experimental design (splits, ablations, etc.) matches `docs/microplastic_gnn_research_specification.md`, Section 14 (TRAIN/VALIDATION/TEST DESIGN) and Section 16-24 (various experiments).
[ ] Primary metrics (MAE, RMSE, R², uncertainty, p-values) are taken directly from `experiments/results/final_summary.json` and `experiments/results/statistical_comparison.json`.
[ ] Key results numbers are accurate to at least two significant figures and match the source JSON files.
[ ] Limitations listed are those documented in `docs/MODEL_CARD.md`, Limitations section and `docs/PRESENTATION_SLIDE_14_LIMITATIONS.md`.
[ ] Conclusions are evidence-based and do not overstate the results (see `docs/PRESENTATION_CLAIM_AUDIT.md`).
[ ] No claim implies causation from prediction (e.g., "X causes Y" is avoided unless supported by specific causal analysis, which is not present).
[ ] No claim of novelty is made without evidence (e.g., "first to..." is avoided unless exhaustive literature search supports it).
[ ] All figures, charts, and maps are generated from actual project data and code (see `docs/PRESENTATION_DESIGN_SPEC.md` for guidance).
[ ] Every numerical value in the presentation can be traced back to a specific line in an experiment result file, model card, or specification document.
[ ] AI usage, if mentioned, matches what is documented in `docs/AI_USAGE_LOG.md`.

## Clarity and Design
[ ] Text is concise; slides do not contain paragraphs of text.
[ ] Fonts are large enough to be read from a distance (minimum 24pt for body, 28pt for headings, 36pt+ for titles).
[ ] Color contrast meets accessibility standards (WCAG AA); text is readable against background.
[ ] All axes are labeled with units and variable names.
[ ] Legends are present where needed and are not overly complex.
[ ] Figures are not cluttered; unnecessary decorations, 3D effects, or excessive gridlines are removed.
[ ] The visual hierarchy guides the viewer to the main message (use size, color, and placement).
[ ] Slide transitions are simple and not distracting.
[ ] Speaker notes are prepared for each slide (see `docs/PRESENTATION_SCRIPT.md` and `docs/PRESENTATION_SCRIPT_3_MIN.md`).
[ ] The presentation can be delivered within the required time limit (check competition rules).
[ ] A 3-minute version is prepared (see `docs/PRESENTATION_SCRIPT_3_MIN.md`).
[ ] A 60-second elevator pitch is prepared (see `docs/ELEVATOR_PITCH.md`).

## Scientific Communication
[ ] The opening clearly states the problem and why it matters.
[ ] The knowledge gap is clearly articulated before presenting the research question.
[ ] The hypothesis is stated in both technical and accessible language.
[ ] The method section focuses on what was actually done, not on hypothetical or planned steps.
[ ] The results section shows both the model performance and baseline performance for fair comparison.
[ ] Error bars, uncertainty bands, or confidence intervals are shown where appropriate.
[ ] The interpretation of results is cautious and acknowledges limitations.
[ ] The conclusion directly answers the research question based on the evidence.
[ ] The student can explain every slide, figure, and number without referring to notes.
[ ] The student is prepared to answer judge questions using evidence from the project (see `docs/PRESENTATION_JUDGE_QA.md`).
[ ] The student knows which claims are strongest, which are tentative, and which are limitations.
[ ] The presentation avoids jargon that is not defined; necessary technical terms are explained simply.
[ ] The overall tone is that of a careful scientist, not a marketer or advocate.

## Competition-Specific Requirements (Verify with Your Fair's Rules)
[ ] If required, the research plan, abstract, logbook, and bibliography are prepared separately (not on the poster).
[ ] If required, the AI usage disclosure is included on the poster or in the presentation (see `docs/AI_USAGE_LOG.md`).
[ ] If required, sources of data and funding are acknowledged.
[ ] If required, the poster size and orientation comply with regulations.
[ ] If required, the presentation length and format (e.g., slides vs. oral) are respected.
[ ] Any required forms (safety, human/animal subjects) are completed and available if needed.

## Reproducibility and Integrity
[ ] The repository includes a `README.md` that explains how to reproduce the main results.
[ ] The `docs/REPRODUCIBILITY_PACKAGE.md` lists software, versions, dependencies, and commands to run.
[ ] Random seeds used in experiments are documented (see `experiments/results/` JSON files for seed information).
[ ] Checkpoints or saved models are available (in `experiments/checkpoints/`).
[ ] The student has run a quick sanity check that the code still runs and produces similar results (if time permits).
[ ] All data used in the project is properly cited and attributed to its source (USGS, etc.).
[ ] No private or sensitive data is included in the presentation or poster.

## Final Verification
[ ] The student has presented the talk to a peer or advisor and incorporated feedback.
[ ] The student has practiced answering questions using the judge question database.
[ ] The student feels confident defending every aspect of the project.
[ ] The student is ready to enjoy the experience and share their scientific work.

If any checkbox is unchecked, return to the relevant source document or revise the presentation/poster accordingly.