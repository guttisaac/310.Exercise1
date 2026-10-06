# DSDA 310 Project 1

This project uses the UCI Adult dataset to predict whether a person's annual income is above $50,000. It compares logistic regression with a random forest, explains the predictions, and checks where the model makes mistakes.


| File | What it contains |
| --- | --- |
| [01_proposal.md](01_proposal.md) | Research question, feature choices, split strategy, and metric justification |
| [notebook.ipynb](notebook.ipynb) | Main notebook with saved results and explanations |
| [analysis.py](analysis.py) | Data cleaning, model training, evaluation, and report generation |
| [results.md](results.md) | Baseline and random forest results on the same split |
| [explainability/README.md](explainability/README.md) | Overall feature importance and one misclassified person's prediction |
| [fairness_leakage_check.md](fairness_leakage_check.md) | Feature-removal tests, group error rates, and limitations |
| [executive_summary.md](executive_summary.md) | Short summary for a nontechnical reader |
| [ai_use_log.md](ai_use_log.md) | AI assistance used and how the work was checked |
| [verify_results.py](verify_results.py) | Checks the saved results against the predictions |

The other notebooks are earlier class exercises and experiments. Use `notebook.ipynb` as the main project notebook.

## Run the project

Python 3.13.0 was used. Keep `DSDA310/adult.data` and `DSDA310/adult.test` in the folder; the analysis needs both files. From this project folder, run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.2.4 pandas==2.2.3 scikit-learn==1.9.1
python analysis.py
python verify_results.py
```

The saved tables can be read without running anything. A new run recreates CSV and JSON outputs in `outputs/`, plus explanation figures and data in `explainability/`. Run the analysis before running `verify_results.py` by itself. Running the analysis also rewrites the results, evaluation, explanation, and executive-summary reports using the text in `analysis.py`.

## Main findings

1. On the separate 16,281-record test file, logistic regression reached 84.9% accuracy and 0.647 F1. The random forest reached 86.5% accuracy and 0.674 F1. The forest performed better overall, but still missed 40.9% of the higher-income records.

2. F1 is the main comparison metric because higher-income records are the minority. Accuracy, precision, recall, and confusion matrices are included so the tradeoffs remain visible. The forest settings were chosen using validation data, with the classification threshold fixed at 0.50.

3. Removing capital gains and losses reduced test F1 from 0.674 to 0.610. This checks a possible prediction-time availability issue; it does not prove those fields are always leakage. Recall also differed across sex and race groups. These are historical patterns in the dataset, not causal conclusions about people.

## Sources and scope

The [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult) contains 48,842 records from a 1994 census database. Credit: Becker and Kohavi, 1996, DOI 10.24432/C5XW20, CC BY 4.0. The local [adult.names](DSDA310/adult.names) describes the fields and original split. Further reading and the reasons for the feature choices are in the proposal.

The current leakage hypothesis was added during completion work, rather than recorded in the original September 3 proposal. Earlier AI-use history was not available. The team still needs to review the work, add any genuine earlier notes it has, and prepare for the October 8 oral defense.
