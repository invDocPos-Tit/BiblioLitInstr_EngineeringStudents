# CiteSpace-style Analysis Report (own implementation)

> **Not produced by CiteSpace.** Own implementation, in this pipeline, of the analyses that CiteSpace popularized, following their published definitions: burst detection (Kleinberg, 2002), betweenness centrality (Freeman, 1977), sigma = (centrality + 1)^burst (Chen et al., 2009), modularity Q and silhouette for cluster quality, and log-likelihood ratio (LLR) cluster labels. It has not been validated against the CiteSpace software; cite the methods, not the tool.

---

## Network Quality

- **Modularity Q** = 0.6540 (significant; Q > 0.3 = significant)
- **Mean Silhouette S** = 0.2082 (weak; S > 0.7 = convincing)

| Cluster | Silhouette | Quality |
|---------|-----------|---------|
| #0 | 0.6667 | reasonable |
| #1 | 0.0095 | weak |
| #2 | 0.6667 | reasonable |
| #3 | 0.6667 | reasonable |
| #4 | 0.1481 | weak |
| #5 | 0.6214 | reasonable |
| #6 | 0.2297 | weak |
| #7 | 0.0776 | weak |
| #8 | 0.0500 | weak |
| #9 | 0.3107 | weak |
| #10 | 0.0804 | weak |
| #11 | 0.6667 | reasonable |
| #12 | 0.1938 | weak |
| #13 | 0.7495 | convincing |
| #14 | 0.6667 | reasonable |
| #15 | 0.2857 | weak |
| #16 | 0.2381 | weak |
| #17 | 0.4127 | weak |
| #18 | 0.3098 | weak |
| #19 | 0.6667 | reasonable |
| #20 | 0.2616 | weak |
| #21 | 0.4441 | weak |
| #22 | 0.6667 | reasonable |
| #23 | 0.5079 | reasonable |
| #24 | 0.6667 | reasonable |
| #25 | 0.3333 | weak |
| #26 | 0.6667 | reasonable |
| #27 | 0.6667 | reasonable |
| #28 | 0.6667 | reasonable |

## 1. Burst Detection (Kleinberg)

### Keywords Bursts

| Entity | Begin | End | Duration | Level | Strength |
|--------|-------|-----|----------|-------|----------|
| education | 2016 | 2017 | 2 | 2 | 12.18 |
| engineering | 2014 | 2016 | 3 | 2 | 11.64 |
| validity | 2015 | 2015 | 1 | 3 | 11.21 |
| reliability | 2014 | 2015 | 2 | 2 | 10.37 |
| survey | 2014 | 2020 | 7 | 1 | 7.72 |
| education computing | 2025 | 2026 | 2 | 1 | 6.16 |
| engineering student | 2015 | 2021 | 7 | 1 | 6.11 |
| artificial intelligence | 2026 | 2026 | 1 | 1 | 5.58 |
| student attitudes | 2021 | 2022 | 2 | 1 | 4.91 |
| undergraduate students | 2014 | 2016 | 3 | 1 | 4.9 |
| instruments | 2014 | 2020 | 7 | 1 | 4.47 |
| generative artificial intelligence | 2025 | 2026 | 2 | 1 | 4.35 |
| design | 2014 | 2021 | 8 | 1 | 4.17 |
| e-learning | 2026 | 2026 | 1 | 1 | 3.58 |
| correlation | 2014 | 2019 | 6 | 1 | 3.53 |
| test construction | 2014 | 2018 | 5 | 1 | 3.47 |
| undergraduate engineering students | 2015 | 2020 | 6 | 1 | 3.25 |
| modeling | 2026 | 2026 | 1 | 1 | 3.2 |
| soft skill | 2021 | 2024 | 4 | 1 | 3.0 |
| technological innovation | 2025 | 2025 | 1 | 1 | 2.99 |

### Authors Bursts

No significant bursts detected.

### References Bursts

| Entity | Begin | End | Duration | Level | Strength |
|--------|-------|-----|----------|-------|----------|
| Nassim Tabri (2012) — Principles and Practice o... | 2021 | 2022 | 2 | 1 | 3.79 |
| Robert W. Lent (1994) — Toward a Unifying Socia... | 2014 | 2022 | 9 | 1 | 3.68 |
| Benjamin S. Bloom (1956) — Taxonomy of Educatio... | 2014 | 2016 | 3 | 1 | 2.95 |
| Cecilia Ka Yuk Chan (2017) — A Validated and Re... | 2020 | 2023 | 4 | 1 | 2.76 |
| Unknown (2009) — Educating the engineer of 2020... | 2014 | 2017 | 4 | 1 | 2.67 |

### Journals Bursts

| Entity | Begin | End | Duration | Level | Strength |
|--------|-------|-----|----------|-------|----------|
| ASEE Annual Conference and Exposition, Conferen... | 2017 | 2020 | 4 | 1 | 6.26 |
| Journal of Engineering Education | 2014 | 2018 | 5 | 1 | 3.1 |

## 2. Pivot Nodes (Inter-cluster Bridges)

| Node | Cluster | Betweenness | Neighbor Clusters | Bridge Score |
|------|---------|------------|-------------------|-------------|
| Li‐tze Hu (1999) — Cutoff criteria for fit inde... | 1 | 0.4065 | 6 | 2.4389 |
| Claes Fornell (1981) — Evaluating Structural Eq... | 1 | 0.1179 | 6 | 0.7075 |
| Albert Bandura (1977) — Self-efficacy: Toward a... | 4 | 0.0770 | 5 | 0.3848 |
| Nassim Tabri (2012) — Principles and Practice o... | 7 | 0.0435 | 7 | 0.3048 |
| Natasha A. Mamaril (2016) — Measuring Undergrad... | 7 | 0.0487 | 6 | 0.2920 |
| Rose M. Marra (2009) — Women Engineering Studen... | 7 | 0.0541 | 5 | 0.2705 |
| Albert Bandura (1999) — Self-Efficacy: The Exer... | 7 | 0.0453 | 5 | 0.2264 |
| Cecilia Ka Yuk Chan (2017) — A Validated and Re... | 4 | 0.0307 | 5 | 0.1537 |
| Adam Robert Carberry (2010) — Measuring Enginee... | 7 | 0.0303 | 4 | 0.1210 |
| Peter M. Bentler (1980) — Significance tests an... | 5 | 0.0266 | 4 | 0.1062 |
| Rex B. Kline (1998) — Principles and Practice o... | 7 | 0.0173 | 5 | 0.0867 |
| Robert W. Lent (1994) — Toward a Unifying Socia... | 7 | 0.0172 | 5 | 0.0858 |
| Unknown (1997) — Self-efficacy: the exercise of... | 6 | 0.0159 | 5 | 0.0797 |
| Raymond B. Cattell (1966) — The Scree Test For ... | 1 | 0.0186 | 4 | 0.0744 |
| Michael K. Ponton (2001) — Understanding the Ro... | 7 | 0.0181 | 4 | 0.0724 |

## 3. Sigma (Σ) Rankings

| Rank | Node | Betweenness | Burstness | Sigma |
|------|------|------------|-----------|-------|
| 1 | Li‐tze Hu (1999) — Cutoff criteria for fit inde... | 0.4065 | 0.00 | 1.4065 |
| 2 | Nassim Tabri (2012) — Principles and Practice o... | 0.0435 | 3.79 | 1.1753 |
| 3 | Claes Fornell (1981) — Evaluating Structural Eq... | 0.1179 | 0.00 | 1.1179 |
| 4 | Cecilia Ka Yuk Chan (2017) — A Validated and Re... | 0.0307 | 2.76 | 1.0871 |
| 5 | Albert Bandura (1977) — Self-efficacy: Toward a... | 0.0770 | 0.00 | 1.0770 |
| 6 | Robert W. Lent (1994) — Toward a Unifying Socia... | 0.0172 | 3.68 | 1.0646 |
| 7 | Rose M. Marra (2009) — Women Engineering Studen... | 0.0541 | 0.00 | 1.0541 |
| 8 | Natasha A. Mamaril (2016) — Measuring Undergrad... | 0.0487 | 0.00 | 1.0487 |
| 9 | Albert Bandura (1999) — Self-Efficacy: The Exer... | 0.0453 | 0.00 | 1.0453 |
| 10 | Unknown (2009) — Educating the engineer of 2020... | 0.0143 | 2.67 | 1.0387 |
| 11 | Adam Robert Carberry (2010) — Measuring Enginee... | 0.0303 | 0.00 | 1.0303 |
| 12 | Peter M. Bentler (1980) — Significance tests an... | 0.0266 | 0.00 | 1.0266 |
| 13 | Unknown (2006) — Experimental Design and Analysis | 0.0224 | 0.00 | 1.0224 |
| 14 | Anna B. Costello (2005) — Best practices in exp... | 0.0219 | 0.00 | 1.0219 |
| 15 | Allison Godwin (2016) — Identity, Critical Agen... | 0.0197 | 0.00 | 1.0197 |
| 16 | Paul R. Pintrich (1993) — Reliability and Predi... | 0.0194 | 0.00 | 1.0194 |
| 17 | Raymond B. Cattell (1966) — The Scree Test For ... | 0.0186 | 0.00 | 1.0186 |
| 18 | Michael K. Ponton (2001) — Understanding the Ro... | 0.0181 | 0.00 | 1.0181 |
| 19 | Rex B. Kline (1998) — Principles and Practice o... | 0.0173 | 0.00 | 1.0173 |
| 20 | Meng‐Jung Tsai (2018) — Developing the Computer... | 0.0164 | 0.00 | 1.0164 |

## 4. Research Fronts

### Cluster 1: engineering education; education; engineering students
- Intellectual base: 160 references
- Research front: 100 articles
- Year range: 2014-2026
- Top keywords: engineering education, education, engineering students, mathematics education, self-efficacy

### Cluster 7: engineering education; self-efficacy; education
- Intellectual base: 169 references
- Research front: 83 articles
- Year range: 2014-2026
- Top keywords: engineering education, self-efficacy, education, engineering students, students

### Cluster 4: engineering education; education; engineering students
- Intellectual base: 102 references
- Research front: 71 articles
- Year range: 2014-2026
- Top keywords: engineering education, education, engineering students, students, self-efficacy

### Cluster 10: engineering education; education; students
- Intellectual base: 25 references
- Research front: 52 articles
- Year range: 2014-2026
- Top keywords: engineering education, education, students, engineering education and pedagogy, engineering students

### Cluster 8: engineering education; education; students
- Intellectual base: 52 references
- Research front: 32 articles
- Year range: 2014-2026
- Top keywords: engineering education, education, students, engineering students, self-efficacy

### Cluster 6: engineering education; self-efficacy; education
- Intellectual base: 33 references
- Research front: 31 articles
- Year range: 2014-2026
- Top keywords: engineering education, self-efficacy, education, engineering students, applied psychology

### Cluster 5: engineering education; engineering students; foreign countries
- Intellectual base: 33 references
- Research front: 17 articles
- Year range: 2016-2026
- Top keywords: engineering education, engineering students, foreign countries, education, factor analysis

### Cluster 9: engineering education; education; engineering identity
- Intellectual base: 18 references
- Research front: 12 articles
- Year range: 2017-2026
- Top keywords: engineering education, education, engineering identity, mathematics education, engineering students

### Cluster 21: engineering education; project-based learning; education
- Intellectual base: 4 references
- Research front: 10 articles
- Year range: 2014-2026
- Top keywords: engineering education, project-based learning, education, student engagement, active learning

### Cluster 12: engineering education; education; mathematics education
- Intellectual base: 10 references
- Research front: 9 articles
- Year range: 2014-2026
- Top keywords: engineering education, education, mathematics education, self-efficacy, teaching and learning programming

### Cluster 18: engineering education; study abroad; test validity
- Intellectual base: 17 references
- Research front: 8 articles
- Year range: 2015-2026
- Top keywords: engineering education, study abroad, test validity, competence, engineering students

### Cluster 13: engineering education; college students; education
- Intellectual base: 2 references
- Research front: 7 articles
- Year range: 2014-2026
- Top keywords: engineering education, college students, education, reliability, higher education

### Cluster 16: engineering education; education; students
- Intellectual base: 8 references
- Research front: 7 articles
- Year range: 2015-2026
- Top keywords: engineering education, education, students, teamwork, scoring rubrics

### Cluster 15: engineering education; education; sustainability
- Intellectual base: 7 references
- Research front: 6 articles
- Year range: 2015-2026
- Top keywords: engineering education, education, sustainability, gender differences, engineering students

### Cluster 20: engineering education; education; peer assessment
- Intellectual base: 9 references
- Research front: 6 articles
- Year range: 2015-2026
- Top keywords: engineering education, education, peer assessment, self-assessment, critical thinking

### Cluster 25: engineering education; project-based learning; education
- Intellectual base: 4 references
- Research front: 6 articles
- Year range: 2024-2026
- Top keywords: engineering education, project-based learning, education, problem and project based learning, artificial intelligence

### Cluster 14: engineering education; student motivation; sustainability
- Intellectual base: 4 references
- Research front: 5 articles
- Year range: 2014-2023
- Top keywords: engineering education, student motivation, sustainability, validity, engineering students

### Cluster 17: engineering education; competence; sustainability
- Intellectual base: 7 references
- Research front: 4 articles
- Year range: 2014-2023
- Top keywords: engineering education, competence, sustainability, student attitudes, engineering students

### Cluster 27: active learning; engineering education; academic achievement
- Intellectual base: 10 references
- Research front: 4 articles
- Year range: 2014-2025
- Top keywords: active learning, engineering education, academic achievement, qualitative research, questionnaires

### Cluster 2: education; mathematics education; civil engineering students
- Intellectual base: 9 references
- Research front: 4 articles
- Year range: 2018-2020
- Top keywords: education, mathematics education, civil engineering students, students, engineering education and curriculum development

### Cluster 19: critical thinking; philosophy; quantum physics
- Intellectual base: 29 references
- Research front: 3 articles
- Year range: 2020-2026
- Top keywords: critical thinking, philosophy, quantum physics, higher education, educational innovation

### Cluster 23: entrepreneurship education; entrepreneurship; engineering education
- Intellectual base: 7 references
- Research front: 3 articles
- Year range: 2018-2025
- Top keywords: entrepreneurship education, entrepreneurship, engineering education, engineering students, education

### Cluster 26: engineering education; statics; curricula
- Intellectual base: 4 references
- Research front: 3 articles
- Year range: 2019-2025
- Top keywords: engineering education, statics, curricula, students, education

### Cluster 11: engineering education; self-efficacy; social cognitive career theory
- Intellectual base: 4 references
- Research front: 2 articles
- Year range: 2021-2025
- Top keywords: engineering education, self-efficacy, social cognitive career theory, scale development, engineering students

### Cluster 22: engineering education; empathy; compassion
- Intellectual base: 4 references
- Research front: 2 articles
- Year range: 2024-2025
- Top keywords: engineering education, empathy, compassion, structural equation modeling, undergraduates

### Cluster 24: graduate students; engineering education; engineering graduate students
- Intellectual base: 24 references
- Research front: 2 articles
- Year range: 2018-2021
- Top keywords: graduate students, engineering education, engineering graduate students, graduate education, writing

### Cluster 28: mathematics education; engineering students; scale development
- Intellectual base: 4 references
- Research front: 2 articles
- Year range: 2023-2026
- Top keywords: mathematics education, engineering students, scale development, stem education, engineering education

### Cluster 0: engineering education; education; engineering design skills
- Intellectual base: 5 references
- Research front: 2 articles
- Year range: 2016-2020
- Top keywords: engineering education, education, engineering design skills, students, knowledge retention

### Cluster 3: engineering education; engineering; sustainability
- Intellectual base: 4 references
- Research front: 2 articles
- Year range: 2014-2016
- Top keywords: engineering education, engineering, sustainability, concept mapping, concept formation

## 5. Keyword Co-occurrence Clusters

| Cluster | Size | Top Keywords |
|---------|------|-------------|
| #0 | 3 | graduate students; engineering graduate students; graduate e |
| #1 | 63 | engineering students; self-efficacy; self efficacy |
| #2 | 75 | engineering education; students; curricula |
| #3 | 24 | higher education; student motivation; motivation |
| #4 | 15 | creativity; learning; human |
| #5 | 20 | education; project-based learning; soft skills |

## 6. Co-citation Cluster Labels

| Cluster | Size | TF-IDF Label | LLR Label |
|---------|------|-------------|-----------|
| 0 | 5 | systems; thinking; modeling | functional; funskill; systems |
| 1 | 160 | self; entrepreneurial; self efficacy | entrepreneurial; literacy; genai |
| 2 | 9 | em; problem; solving | psi; solving; problem |
| 3 | 4 | scoring; concept; maps | scoring; concept; maps |
| 4 | 102 | self efficacy; self; efficacy | cdio; skills; mind |
| 5 | 33 | self efficacy; efficacy; factor | internship; generic; ers |
| 6 | 33 | self efficacy; self; efficacy | skills; agency; computational |
| 7 | 169 | self efficacy; self; efficacy | efficacy; self; nursing |
| 8 | 52 | self; self efficacy; efficacy | architectural; lsp; grit |
| 9 | 18 | identity; engineering identity; fgcs | identity; engineering; math |
| 10 | 25 | learning; self; skills | entrepreneurial; hazard; manufacturing |
| 11 | 4 | pbl; sections; learning experiences | pbl; sections; experiences |
| 12 | 10 | programming; computational; self efficac | programming; microelectronics; computati |
| 13 | 2 | se; self efficacy; cultural | disposition; risk; cultural |
| 14 | 4 | se; inf; ic | inf; bmt; tlo |
| 15 | 7 | sustainability; se; learning | sustainability; environmental; sustainab |
| 16 | 8 | rubric; analytic; skills | rubric; analytic; forums |
| 17 | 7 | contextual; contextual competence; susta | contextual; sustainability; competence |
| 18 | 17 | cultural; global; team | cultural; sjt; global |
| 19 | 29 | model; synthesis; critical thinking | usepackage; philosophical; session |
| 20 | 9 | peer; assessment; peer self | peer; assessment; videos |
| 21 | 4 | pbl; pjbl; learning | pjbl; pbl; engagement |
| 22 | 4 | compassion; empathy; engagement | compassion; empathetic; empathize |
| 23 | 7 | entrepreneurial; entrepreneurship; motiv | entrepreneurial; entrepreneurship; indir |
| 24 | 24 | writing; graduate; attitudes | writing; graduate; writers |
| 25 | 4 | ic pbl; pbl; ic | platrob; pbl; robotics |
| 26 | 4 | statics; learning; intervention | representations; statics; vector |
| 27 | 10 | learning; peer; pjbl | peer; pjbl; mechanics |
| 28 | 4 | mathematical; techno; literacy | mathematical; techno; tml |

### Cluster 0 Detail
- Size: 5 references, 2 citing articles
- **TF-IDF terms:** systems, thinking, modeling, engineering design, scoring
- **LLR terms:** functional, funskill, systems, function, modeling
- **MI terms:** functional, improves, function, generated, longitudinal

### Cluster 1 Detail
- Size: 160 references, 100 citing articles
- **TF-IDF terms:** self, entrepreneurial, self efficacy, efficacy, skills
- **LLR terms:** entrepreneurial, literacy, genai, psycap, intention
- **MI terms:** acceptance, shaped, tpb, aforementioned, media

### Cluster 2 Detail
- Size: 9 references, 4 citing articles
- **TF-IDF terms:** em, problem, solving, problem solving, writing
- **LLR terms:** psi, solving, problem, capstone, ability
- **MI terms:** parents, psi, gpa, style, avoidance

### Cluster 3 Detail
- Size: 4 references, 2 citing articles
- **TF-IDF terms:** scoring, concept, maps, judges, concept maps
- **LLR terms:** scoring, concept, maps, judges, methods
- **MI terms:** knowledge, work, three, student, engineering

### Cluster 4 Detail
- Size: 102 references, 71 citing articles
- **TF-IDF terms:** self efficacy, self, efficacy, skills, learning
- **LLR terms:** cdio, skills, mind, neurodiverse, innovation
- **MI terms:** conceive, emergent, grammar, received, lms

### Cluster 5 Detail
- Size: 33 references, 17 citing articles
- **TF-IDF terms:** self efficacy, efficacy, factor, self, internship
- **LLR terms:** internship, generic, ers, competencies, strong
- **MI terms:** polytechnic, paramount, weq, hong, internship

### Cluster 6 Detail
- Size: 33 references, 31 citing articles
- **TF-IDF terms:** self efficacy, self, efficacy, skills, factor
- **LLR terms:** skills, agency, computational, computing, factor
- **MI terms:** lives, portugal, concern, place, negatively

### Cluster 7 Detail
- Size: 169 references, 83 citing articles
- **TF-IDF terms:** self efficacy, self, efficacy, learning, scale
- **LLR terms:** efficacy, self, nursing, cad, competencies
- **MI terms:** worksheet, event, watch, formulate, labs

### Cluster 8 Detail
- Size: 52 references, 32 citing articles
- **TF-IDF terms:** self, self efficacy, efficacy, learning, scale
- **LLR terms:** architectural, lsp, grit, motivation, mslq
- **MI terms:** srl, studios, belief, perseverance, grit

### Cluster 9 Detail
- Size: 18 references, 12 citing articles
- **TF-IDF terms:** identity, engineering identity, fgcs, graduate, graduate students
- **LLR terms:** identity, engineering, math, graduate, fgcs
- **MI terms:** doctoral, initiated, wherein, math, attempting

### Cluster 10 Detail
- Size: 25 references, 52 citing articles
- **TF-IDF terms:** learning, self, skills, self efficacy, efficacy
- **LLR terms:** entrepreneurial, hazard, manufacturing, how, mse
- **MI terms:** additive, positioned, methodologies, staged, instituto

### Cluster 11 Detail
- Size: 4 references, 2 citing articles
- **TF-IDF terms:** pbl, sections, learning experiences, learning, experiences
- **LLR terms:** pbl, sections, experiences, eles, baseline
- **MI terms:** scores, evidence, four, future, scale

### Cluster 12 Detail
- Size: 10 references, 9 citing articles
- **TF-IDF terms:** programming, computational, self efficacy, self, efficacy
- **LLR terms:** programming, microelectronics, computational, algorithmic, cpses
- **MI terms:** cpses, surrounding, code, computational, scaffold

### Cluster 13 Detail
- Size: 2 references, 7 citing articles
- **TF-IDF terms:** se, self efficacy, cultural, efficacy, risk
- **LLR terms:** disposition, risk, cultural, affect, tutoring
- **MI terms:** century, considered, explored, allow, challenge

### Cluster 14 Detail
- Size: 4 references, 5 citing articles
- **TF-IDF terms:** se, inf, ic, learning, sustainability
- **LLR terms:** inf, bmt, tlo, abroad, affect
- **MI terms:** target, variety, motivated, affect, demonstrate

### Cluster 15 Detail
- Size: 7 references, 6 citing articles
- **TF-IDF terms:** sustainability, se, learning, environmental, sustainable
- **LLR terms:** sustainability, environmental, sustainable, hydrogen, dbl
- **MI terms:** demand, sustainability, effort, promoting, sustainable

### Cluster 16 Detail
- Size: 8 references, 7 citing articles
- **TF-IDF terms:** rubric, analytic, skills, process, rubrics
- **LLR terms:** rubric, analytic, forums, rubrics, process
- **MI terms:** raters, analytic, rater, alongside, holistic

### Cluster 17 Detail
- Size: 7 references, 4 citing articles
- **TF-IDF terms:** contextual, contextual competence, sustainability, competence, scale
- **LLR terms:** contextual, sustainability, competence, extracurricular, interaction
- **MI terms:** described, considered, contextual, abet, meet

### Cluster 18 Detail
- Size: 17 references, 8 citing articles
- **TF-IDF terms:** cultural, global, team, abroad, teams
- **LLR terms:** cultural, sjt, global, abroad, logistics
- **MI terms:** intercultural, abroad, diversity, cultural, similarities

### Cluster 19 Detail
- Size: 29 references, 3 citing articles
- **TF-IDF terms:** model, synthesis, critical thinking, test, physics
- **LLR terms:** usepackage, philosophical, session, quantum, installation
- **MI terms:** quantum, philosophical, composed, ten, cohorts

### Cluster 20 Detail
- Size: 9 references, 6 citing articles
- **TF-IDF terms:** peer, assessment, peer self, self, learning
- **LLR terms:** peer, assessment, videos, exit, electric
- **MI terms:** routines, exams, tended, workload, standards

### Cluster 21 Detail
- Size: 4 references, 10 citing articles
- **TF-IDF terms:** pbl, pjbl, learning, ic pbl, ic
- **LLR terms:** pjbl, pbl, engagement, learning, digital
- **MI terms:** absence, pbl, pjbl, actionable, methodological

### Cluster 22 Detail
- Size: 4 references, 2 citing articles
- **TF-IDF terms:** compassion, empathy, engagement, identity, coursework
- **LLR terms:** compassion, empathetic, empathize, humanities, ceas
- **MI terms:** empathy, correlations, identity, completed, undergraduates

### Cluster 23 Detail
- Size: 7 references, 3 citing articles
- **TF-IDF terms:** entrepreneurial, entrepreneurship, motivation, indirect, online
- **LLR terms:** entrepreneurial, entrepreneurship, indirect, person, motivation
- **MI terms:** develops, resilience, mediation, entrepreneurship, useful

### Cluster 24 Detail
- Size: 24 references, 2 citing articles
- **TF-IDF terms:** writing, graduate, attitudes, graduate students, engineering graduate
- **LLR terms:** writing, graduate, writers, rhetorical, patterns
- **MI terms:** calculated, writers, perfectionism, procrastination, affective

### Cluster 25 Detail
- Size: 4 references, 6 citing articles
- **TF-IDF terms:** ic pbl, pbl, ic, learning, ai
- **LLR terms:** platrob, pbl, robotics, modular, sdgs
- **MI terms:** observations, autonomous, assisted, designs, pjbl

### Cluster 26 Detail
- Size: 4 references, 3 citing articles
- **TF-IDF terms:** statics, learning, intervention, exam, use
- **LLR terms:** representations, statics, vector, representational, prototype
- **MI terms:** calculation, difficulty, statics, interested, commonly

### Cluster 27 Detail
- Size: 10 references, 4 citing articles
- **TF-IDF terms:** learning, peer, pjbl, time, group
- **LLR terms:** peer, pjbl, mechanics, learning, character
- **MI terms:** character, avoidance, mechanics, centered, quasi

### Cluster 28 Detail
- Size: 4 references, 2 citing articles
- **TF-IDF terms:** mathematical, techno, literacy, perceived, pre
- **LLR terms:** mathematical, techno, tml, literacy, tmls
- **MI terms:** techno, randomly, mathematical, designing, because

---
*Generated by the pipeline (Phase 2h): CiteSpace-style analyses, own implementation; not the CiteSpace software.*