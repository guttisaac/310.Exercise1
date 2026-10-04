# AI use log

| Task | What happened to the AI output | How it was checked |
| --- | --- | --- |
| Review the assignment and existing files | Used to identify missing work | Compared the assignment's milestone list with the proposal, notebooks, and reports in the folder. |
| Draft the idea for analysis | Ran every main-notebook code cell in order. Checked row counts, labels, split separation, and model convergence. |
| Explain the forest | Added global permutation importance and an individual tree-path explanation | Confirmed that the individual's feature contributions plus the starting probability equal the predicted probability. Repeated this check for three other records. |
| Fix a scoring inconsistency | Changed the initial feature-importance scorer | The first draft used scikit-learn's standard F1 scorer, while the project classified probability >=0.50 as positive. At exactly 0.50, the model's default prediction rule can differ. Replaced the scorer with the same threshold rule used elsewhere and tested the tie using a balanced dummy classifier. |
| Check leakage and group errors | Added models without capital gains/losses and without sex/race, plus subgroup tables | Used the same training records and fixed forest settings. Recalculated group counts and error rates from exported predictions. |
| Draft reports and summary | Added Markdown drafts based on the measured results | Compared the saved result tables with a fresh run. Checked source descriptions against UCI and the local dataset notes. |
| Clean the folder | Removed duplicate documents after saving a backup | Checked that code, data, saved notebook outputs, and explanation tables remained. 
| Check submission readiness | All cells passed, and the saved result tables matched the new calculations. |
| Restore the required files and improve readability | Added this log and the README; revised variable names, code formatting, and notebook wording | Reran the revised notebook and verification checks, and compared its numerical tables with the previous version. |

The threshold issue was a real inconsistency in the initial AI draft. The small regression check demonstrates the problem; it does not show that any original Adult prediction was exactly 0.50. The corrected analysis was rerun before the results were saved.