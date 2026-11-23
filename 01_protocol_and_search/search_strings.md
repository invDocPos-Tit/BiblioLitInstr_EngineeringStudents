# Supplementary material A. Search strings

Search strings used, as reported by the authors. All share the same four blocks of terms; the syntax and limits of each database differ.

*Scopus*

```
TITLE-ABS-KEY (
  ( "engineering education" OR "engineering student*" OR "engineering undergraduate*" OR "engineering curriculum" OR "electronic* engineering" OR "electrical engineering" )
  AND
  ( competenc* OR "technical skill*" OR "technical competenc*" OR "engineering skill*" OR "engineering competenc*" OR "soft skill*" OR "transversal skill*" OR "generic skill*" OR "professional skill*" OR "employability skill*" OR "self-efficacy" OR "critical thinking" OR "problem-solving skill*" OR "design skill*" OR "communication skill*" OR "teamwork skill*" )
  AND
  ( questionnaire* OR scale* OR inventory OR rubric* OR "measurement instrument*" OR "assessment tool*" OR "assessment instrument*" )
  AND
  ( psychometric* OR validit* OR validat* OR reliabilit* OR "internal consistency" OR "factor analys*" OR "item response theory" OR "measurement invariance" )
)
AND PUBYEAR > 2013
AND PUBYEAR < 2027
```

*Web of Science*: the same blocks of terms, with the prefix `TS=(` and the limit `AND PY=(2014-2026)`.

*EBSCOhost (Education Source, Engineering Source, ERIC and MEDLINE) and Dimensions*: the same blocks of terms, in parentheses, with the period 2014–2026 set in the interface.

*IEEE Xplore* (period 2014–2026 set in the interface)

```
("All Metadata":"engineering education" OR "All Metadata":"engineering student*" OR "All Metadata":"engineering undergraduate*" OR "All Metadata":"engineering curriculum" OR "All Metadata":"electronic engineering" OR "All Metadata":"electronics engineering" OR "All Metadata":"electrical engineering") AND ("All Metadata":competenc* OR "All Metadata":"technical skill" OR "All Metadata":"technical competenc" OR "All Metadata":"engineering skill" OR "All Metadata":"engineering competenc" OR "All Metadata":"soft skill" OR "All Metadata":"transversal skill" OR "All Metadata":"generic skill" OR "All Metadata":"professional skill" OR "All Metadata":"employability skill" OR "All Metadata":"self-efficacy" OR "All Metadata":"critical thinking" OR "All Metadata":"problem-solving skill" OR "All Metadata":"design skill" OR "All Metadata":"communication skill" OR "All Metadata":"teamwork skill") AND ("All Metadata":questionnaire* OR "All Metadata":scale* OR "All Metadata":inventory OR "All Metadata":rubric* OR "All Metadata":"measurement instruments" OR "All Metadata":"measurement instrument" OR "All Metadata":"assessment tools" OR "All Metadata":"assessment tool" OR "All Metadata":"assessment instruments" OR "All Metadata":"assessment instrument") AND ("All Metadata":psychometric* OR "All Metadata":validity OR "All Metadata":validation OR "All Metadata":validate OR "All Metadata":validated OR "All Metadata":validating OR "All Metadata":reliability OR "All Metadata":"internal consistency" OR "All Metadata":"factor analysis" OR "All Metadata":"factor analyses" OR "All Metadata":"item response theory" OR "All Metadata":"measurement invariance")
```

# Search for antecedents (Scopus)
Searches exported on 21 November 2026 (`Export Date` field of the records). Totals are the records exported for each string. Records were merged by DOI or title: 684 unique records. Titles and abstracts were reviewed by the authors, with automatic markers for bibliometrics, review, instrument or competency, and engineering; this was not a paired screening.

| Code | Records | String |
|---|---|---|
| Q10a | 403 | `TITLE-ABS-KEY ( ( bibliometric* OR scientometric* OR "science mapping" OR bibliometrix OR vosviewer ) AND ( "engineering education" OR "engineering student*" OR "engineering curricul*" ) ) AND PUBYEAR > 2014 AND DOCTYPE ( ar OR re )` |
| Q10b | 216 | `TITLE-ABS-KEY ( ( bibliometric* OR scientometric* OR "science mapping" OR "systematic review" OR "literature review" ) AND ( "engineering education" OR "engineering student*" ) ) AND SRCTITLE ( "journal of engineering education" OR "european journal of engineering education" OR "ieee transactions on education" OR "international journal of engineering education" OR "education sciences" ) AND PUBYEAR > 2014` |
| Q15 | 42 | `TITLE-ABS-KEY ( ( "engineering education" OR "engineering student*" ) AND ( competenc* OR "soft skill*" OR "transversal skill*" OR "generic skill*" OR "professional skill*" ) AND ( instrument* OR questionnaire* OR scale* OR "assessment tool*" ) AND ( "systematic review" OR "scoping review" OR "literature review" OR "review of" ) ) AND DOCTYPE ( ar OR re )` |
| Q16 | 7 | `TITLE-ABS-KEY ( ( bibliometric* OR scientometric* ) AND ( "scale development" OR questionnaire* OR "measurement instrument*" OR psychometric* OR "assessment tool*" ) ) AND DOCTYPE ( ar OR re )` |
| Q18 | 88 | `TITLE-ABS-KEY ( ( bibliometric* OR "science mapping" OR scientometric* ) AND ( "self-efficacy" OR "soft skill*" OR "transversal skill*" OR "generic skill*" OR competenc* ) AND ( "engineering education" OR "engineering student*" ) )` |
| Q21 | 9 | `TITLE-ABS-KEY ( "engineering education" AND ( "global south" OR "international collaboration" OR "geographic*" OR "regional distribution" ) AND ( bibliometric* OR "research landscape" OR "science mapping" ) )` |
