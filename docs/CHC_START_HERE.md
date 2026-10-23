# CHC V2 college project release

Start with `CHC_COLLEGE_RESEARCH_PAPER.pdf`, the six-page research manuscript. Its editable text is `CHC_COLLEGE_RESEARCH_PAPER.md`. Add your real name, institution and supervisor, then apply your college's required template before formal submission. No publication or human-review status is claimed.

`CHC_V2_RESEARCH_REPORT.html` and its Markdown source explain the architecture, code changes, experimental results, novelty boundary, hardware, reproduction commands and known failures. `chc_v2/docs/COMPLETE_CODE_EXPLANATION.md` explains modules and functions; `VIVA_GUIDE.md` provides oral-examination questions and answers.

For a presentation, open `CHC_V2_DEMO.html`. It contains ten real held-out recorded examples in their original order. It does not perform new inference. Show a source passage, its four-way evidence status, the chosen verification action and the actual changed continuation. Describe model labels as automatic proxies with human review pending.

The working project is at `C:/Users/BroCode/Documents/nlphallucination/nlphallucination`, branch `codex/chc-research-v2`. Run `./run_chc_v2.ps1` from PowerShell in that folder; open the app address printed in the terminal and append `/chc/propagate`. Existing root verification and `/chc` interfaces remain available. Real live inference requires cached external models and can be slow. An operational live test exposed incorrect answers and event-confusion errors; it is not a validated factual assistant.

The archive contains source, source-row data, trained V2 policies, predictions, reports, tests, notices and audit records. External transformer weights, the original RAGTruth corpus and historical model dependencies are not bundled. They remain in the existing local project/caches; reproduction on a new machine needs the corresponding downloads and original detector/repair checkpoints. Follow `requirements_chc_v2.txt` and the report's commands.

What you can defend: an implemented protocol that learns signed downstream factual-error changes, keeps neutral/negative outcomes, separates unknown evidence, and compares matched policies. Results: 300 attempted questions, 90 primary episodes, 16 final test questions, 31 saved policy checkpoints and 109 passing tests. Same-cohort detector calibration improves F1 .2591 to .3415. One-action held-out answer accuracy stays 12.5%; graph/semantic complexity does not improve the result. Do not claim state-of-the-art performance or proved causal structure.

Remaining scientific work is clearly separated from this completed college project release: independent human review, broader testing and faithful close-prior baselines. Browser visual QA was blocked by localhost isolation and automatic review of file URLs. The six-page PDF was rendered and visually inspected. Word output was withheld because its required renderer was unavailable.
