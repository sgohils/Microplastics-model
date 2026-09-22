# AI Usage Log

## Tools Used
- **Kilo (AI coding assistant)** — Used for code exploration, debugging, documentation generation, and analysis
- **Claude / GPT models** (via Kilo) — Used for technical explanations, code review, and documentation assistance

## Usage by Category

### Coding Assistance
- Exploring repository structure and understanding existing code
- Debugging training scripts and data pipeline issues
- Writing and editing Python code in `src/` modules
- Creating configuration files (config.yaml, mkdocs.yml)

### Documentation Generation
- Generating MODEL_CARD.md from actual experimental results
- Creating science fair documentation templates
- Writing technical explanations for glossary and judge Q&A
- Summarizing experimental results from JSON output files

### Technical Explanation & Brainstorming
- Explaining GNN architecture, GraphSAGE, GRU mechanisms
- Clarifying statistical concepts (RMSE, MAE, R², permutation tests, bootstrap CI)
- Brainstorming experiment designs and ablation strategies
- Organizing literature review notes from actual sources

### Data Processing Assistance
- Understanding data pipeline steps from code
- Interpreting feature engineering choices
- Validating graph construction logic

### Visualization Assistance
- Describing figure concepts for graph explanation, GNN architecture, pipeline diagrams
- Recommending poster figure hierarchy based on actual results

## What the Student Independently Verified
- All experimental results read directly from `experiments/results/*.json` files
- Model architecture confirmed from `src/models/temporal_gnn.py`
- Data sources confirmed from `data/DATA_SOURCES.md` and `data/data_manifest.csv`
- Configuration confirmed from `config.yaml`
- Statistical results confirmed from `experiments/results/statistical_comparison.json`
- All citations and references traced to actual papers/documents in the repository

## What AI Contributed vs. Student Work
| Component | AI Contribution | Student Responsibility |
|-----------|----------------|------------------------|
| Code implementation | Debugging, exploration, some writing | Architecture design, training, experimentation |
| Experimental results | Reading/summarizing JSON outputs | Running experiments, interpreting results |
| Model card | Template structure, formatting | All numbers, conclusions, limitations |
| Science fair docs | Templates, explanations, organization | Competition research, final content, presentation |
| Literature review | Organizing notes from actual sources | Reading papers, selecting relevant work |
| Judge Q&A | Generating questions from results | Preparing answers, practicing delivery |

## Approximate Timeline
- **September 2026**: Code exploration, debugging, initial documentation
- **September 2026**: Model card creation, science fair package generation

## Disclosure Statement
This project used AI assistance (Kilo/Claude) as a research tool for coding, debugging, documentation, and technical explanation. All experimental work, scientific decisions, data analysis, and conclusions were performed and verified by the student. No experimental results, citations, or scientific claims were fabricated by AI. The final presentation, abstract, and competition materials are the student's own work.