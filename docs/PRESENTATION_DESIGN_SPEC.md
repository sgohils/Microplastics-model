# Presentation Design Specification

## Overall aesthetic
* modern scientific
* clean
* high-tech but credible
* environmental
* professional
* competition-ready
* minimal clutter

Avoid:
* excessive gradients
* generic AI imagery
* cheesy stock photos
* giant paragraphs
* excessive animations
* unnecessary icons
* fake 3D graphics
* decorative elements that distract from data

The presentation should feel like:
**scientific research + advanced computational modeling**
rather than:
**startup pitch deck + AI marketing presentation**

## Color System
Use a restrained palette.

### Background
White or very light neutral.

### Primary
Deep blue/green environmental tone (e.g., #003366 or #006400).

### Secondary
Muted teal/blue (e.g., #008080 or #4682B4).

### Accent
One contrasting color used sparingly for:
* model predictions
* important results
* key numbers
(e.g., #FF4500 orange-red or #FFD700 gold)

### Text
Near-black/dark gray (e.g., #2F2F2F).

Do not use more than approximately 4–5 major colors.

Ensure accessibility and strong contrast (WCAG AA compliant).

## Typography
Use a clean modern sans-serif font.

Prioritize:
* readability
* clear hierarchy
* large titles
* short body text
* consistent numbers
* consistent labels

Suggested hierarchy (adjust based on slide size and readability):
Title:
36–48 pt

Section heading:
28–36 pt

Body:
20–28 pt

Chart labels:
18–24 pt

Do not make charts unreadably small just to fit more information.

## Figures to Create
Create or specify the following figures where the data supports them.

### Figure 1
Study-region map
Show Delaware River Basin with monitoring locations, Great Lakes tributaries, and Northeastern U.S. streams.

### Figure 2
River-network graph representation
Directed graph with nodes as river segments, edges as downstream flow, edge thickness representing flow accumulation.

### Figure 3
Data pipeline
Illustrate integration of microplastic observations, river network, hydrology, meteorology, and geography data into a unified graph dataset.

### Figure 4
GNN architecture
As described in Slide 7: feature encoder → GRU → GraphSAGE → prediction head.

### Figure 5
Model comparison
Bar chart showing MAE/RMSE for GNN and baselines (XGBoost, Random Forest, Ridge, MLP, Mean/Median).

### Figure 6
Ablation study
Bar chart showing MAE/RMSE for ablation models (A, B, C, D, E).

### Figure 7
Generalization experiment
Bar chart showing MAE/RMSE for random split, spatial holdout, temporal holdout, spatiotemporal holdout.

### Figure 8
Prediction vs actual
Scatter plot of predicted vs. observed microplastic concentration for test set.

### Figure 9
Residual/error analysis
Histogram of residuals or scatter plot of predicted vs. residual.

### Figure 10
Feature importance
Bar chart showing top features by permutation importance or SHAP values (for baselines) or conceptual upstream influence diagram for GNN.

### Figure 11
Prediction map
Map showing predicted microplastic concentration across the river network (if spatial predictions are made).

### Figure 12
Uncertainty map
Map showing prediction uncertainty (standard deviation) across the river network.

Only create figures that can be generated from actual project data.

## Chart Rules
Every chart must:
* have a clear title
* label axes
* include units where appropriate
* identify train/validation/test where relevant
* include uncertainty/error bars when appropriate
* use consistent scales for comparisons
* avoid misleading axis truncation
* avoid unnecessary 3D
* avoid decorative effects
* use accessible colors
* include sample size where useful

Never manipulate axes to make a small improvement look dramatic.

## Map Rules
Maps must include:
* geographic context
* legend
* scale where appropriate
* north arrow if appropriate
* meaningful color scale
* clear distinction between observed and predicted values
* consistent projection
* data-source attribution where required

Do not create visually impressive maps that obscure uncertainty.

## Results Language Rules
Use precise scientific language.

Preferred:
* "The model achieved…"
* "The experiment showed…"
* "Performance decreased under…"
* "The results suggest…"
* "This pattern was consistent across…"

Avoid:
* "The AI proved…"
* "The AI solved…"
* "The model understands rivers."
* "This proves microplastics move because…"
* "The GNN discovered the cause…"
* "This will solve pollution."

Never turn prediction into causation.