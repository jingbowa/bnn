# Theory and the line of development

BNN averages nearest-neighbor responses over all size-s subsets drawn without
replacement; its exact rank representation is in [algorithms.md](algorithms.md).
The construction builds on Steele (2009) and Biau, Cérou, and Guyader (2010).

Demirkaya, Fan, Gao, Lv, Vossler, and Wang (2024) analyze base DNN and two-scale
bias correction. Use the paper's formal moment, smoothness, design, and
scale-growth conditions. The earlier Fan, Lv, and Wang paper circulated in 2018.

| Topic | Formal source |
|---|---|
| Exact DNN representation | JASA Section 2, equations 5–6 |
| Base bias expansion and distribution theory | JASA Section 3 |
| Two-scale correction and distribution theory | JASA Section 3, equations 9–11 |
| Jackknife and bootstrap variance | JASA Section 4 |
| Bootstrap distribution and treatment effects | JASA supplement |
| Structural elasticity and endogeneity | Wang and Huang, main paper and appendices |

Papers and bibliography are in [references.bib](references.bib). Two-scale
correction is one development in the broader nearest-neighbor construction.
Wang and Huang add first-stage/structural identification conditions to the
regression estimation problem; numerical accuracy does not establish them.

Regional control matters for derivative calculations. Uniform regression
error r_n generally contributes finite-difference error of order r_n/h_n when
the span h_n shrinks. Derivative proofs need compatible rates or a sharper
increment bound, with smoothness and boundary/support control. Uniform
consistency alone does not supply a derivative rate. Supplementary notes
should state their assumptions and proof version alongside the result.

The papers contain the formal statements and proofs. Supplementary theory notes
can be added with versioned assumptions and sources as this resource develops.
