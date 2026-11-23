# Data and code for: Bibliometric profile of the literature on instruments for measuring engineering students' competencies, 2014–2026

Segundo Francisco Segura Altamirano (sseguraal@ucvvirtual.edu.pe) and Bertila Hernández Fernández (hfernandezb@ucvvirtual.edu.pe), Universidad César Vallejo, Chiclayo, Peru.
Deposited in Zenodo. DOI: [to be assigned by Zenodo]. Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), for all the files in this repository.

This repository contains the material needed to check every figure and table of the article: the protocol and search strings, the screening outcome of each record, the identification of the 347 documents, the outputs of the bibliometric analysis tool, a snapshot of the tool's source code, the derived variables and scripts used for the term analysis and the association models, and the figures and tables.

## What is not included, and why

- **Abstracts, keywords and full bibliographic records** exported from Scopus and Web of Science: their terms of use do not allow redistribution. The documents are identified by DOI and title, and the variables derived from their titles, abstracts and keywords are provided (`06_derived_data/document_variables_347.csv`).
- **The tool's data cache and virtual environment**: they contain the licensed records and are not needed to inspect the analysis. Credentials and personal e-mail addresses were removed from the copied files.

## Structure

| Folder | Content |
|---|---|
| `01_protocol_and_search/` | Protocol v3.1 with the four-block search string; screening criteria A1–A7 (A1–A6 were applied in this study); search strings used in each database and for the antecedents; search execution log and export inventory |
| `02_screening/` | `screening_outcomes_1083.csv`: outcome of each screened record (met A1–A6, or excluded with its reason) and whether it is in the final corpus; `prisma_counts.csv`: the counts of Figure 1 |
| `03_corpus/` | `corpus_347.csv`: key, DOI, title, year, document type, source, language, citations, open-access status, SJR quartile and source database of each document |
| `04_tool_outputs/` | Outputs of the analysis tool used in the article: descriptive and supplementary reports, normalised impact (Leiden), bootstrap intervals, reference spectroscopy, bursts and other temporal indicators, thematic map, networks (GraphML), VOSviewer map, run manifest, and the external concordance report with bibliometrix and VOSviewer |
| `05_analysis_tool/` | Source code of the deterministic bibliometric analysis tool used for the article (biblio1403, version 0.1.0), with its documentation |
| `06_derived_data/` | Scripts that read the tool outputs and the tables they produced: term analysis of the search-string blocks, deduplicated reference spectroscopy and sensitivity checks, association models M1–M5, comparison by article processing charge and coverage check against Zhan et al. (2026); `document_variables_347.csv` and `reproduce_models.py`, which refits M1–M5 from that table alone |
| `07_figures_and_tables/` | Figures 1–7, S1 and S2 (PNG and PDF), their captions and the data of Table 1 |
| `SHA256SUMS.txt` | Checksums of every file |

## Where each part of the article comes from

| Article | Files |
|---|---|
| Section 2.2 and Supplementary material A (search) | `01_protocol_and_search/` |
| Section 2.3 and Figure 1 (selection) | `02_screening/` (outcome, eligibility route and exclusion cause of each of the 1,083 screened records); `06_derived_data/calculos_cribado.json` (human-pilot and AI-round agreement, routes and causes) |
| Section 2.2 (block 1 check: electronic and electrical engineering) | `06_derived_data/calculos_bloque1.json` |
| Section 3.1 and Figure 2 (production, channels, databases) | `04_tool_outputs/reports/descriptive_report.md`, `03_corpus/corpus_347.csv`; M1 in `06_derived_data/calculos_modelos.json`; composition of the 2025–2026 increase in `06_derived_data/calculos_salto_2025.json` |
| Sections 3.2–3.3 and Figure 3 (competencies and instrument types) | `06_derived_data/calculos_bloques_cadena.json` (dictionaries and per-document families); M2 and M3 in `calculos_modelos.json` |
| Section 3.4, Table 1 and Figure 4 (measurement-property vocabulary and cited base); Supplementary material D5 and Figure S3 (networks and RPYS) | `calculos_bloques_cadena.json`, `calculos_entrega_en.json` (deduplicated RPYS), `04_tool_outputs/networks/cocitation.graphml`, `04_tool_outputs/spectroscopy/`; M4 in `calculos_modelos.json`; `07_figures_and_tables/table1_most_cited_works.csv` |
| Section 3.5, Figures 5, 6 and 7 (countries, collaboration, visibility) | `06_derived_data/calculos_fwci_campo.json` (FWCI by field, Figure 7; check against the mixed MNCS), `04_tool_outputs/leiden/`, `04_tool_outputs/reproducibility/bootstrap_ci.md`, `04_tool_outputs/networks/coauthorship_countries.graphml`; M5 in `calculos_modelos.json` |
| Section 2.5 (reading of the measurement-property families as sources of validity evidence) and Discussion | `06_derived_data/calculos_fuentes_validez.py` and `calculos_fuentes_validez.json` (regrouping of the per-document families of `calculos_bloques_cadena.json`; no new term matching) |
| Section 3.6 (coverage and sensitivity checks) | `06_derived_data/coverage_check_Zhan2026.csv`, `calculos_entrega_en.json` |
| Section 2.6 and Supplementary material D6 (external concordance) | `04_tool_outputs/external_concordance/` |
| Supplementary material D (dictionaries and additional analyses) | `calculos_bloques_cadena.py`, `04_tool_outputs/networks/bridge_articles_report.json`, `04_tool_outputs/citespace/`, `04_tool_outputs/impact_evolution/`, `04_tool_outputs/reports/strategic_thematic_report.md`, `04_tool_outputs/vosviewer/` |
| Supplementary material E (countries, authors, citation indicators, article processing charges) | `04_tool_outputs/reports/`, `06_derived_data/apc_estratificado.csv` |
| Supplementary material F, Table S1 (association models) | `06_derived_data/calculos_modelos.json` (`model_details`), `tabla_modelos.py`, `tabla_s1_modelos_en.md`, `reproduce_models.py` |

## Reproducing the association models

`python 06_derived_data/reproduce_models.py` (pandas, numpy and statsmodels 0.14.6) refits M1–M5 and M4b from `document_variables_347.csv` and prints the estimates reported in the article. The other scripts in `06_derived_data/` document how the derived data were produced; they read the tool's outputs and its data cache, which is not distributed, so their paths point to the authors' installation.
