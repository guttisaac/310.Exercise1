# Adult income prediction executive summary draft

## Question and approach

Can census characteristics distinguish people earning above $50,000 per year from those earning at or below that amount? We analyzed 48,842 historical Adult dataset records, with one record per individual. The original development file was divided into training and validation groups, while the separate 16,281-record test file measured final performance. Missing entries were filled using training-data values. We compared a simple majority-class rule, logistic regression and a random forest; model settings were chosen using validation data.

## Main findings

On the final test file, logistic regression reached 84.9% accuracy and 0.647 F1, while the random forest reached 86.5% accuracy and 0.674 F1. F1 balances correct positive predictions with detection of higher-income individuals. We emphasized it because a model can achieve high accuracy by predicting the much more common lower-income class.

For the forest, 78.4% of higher-income predictions were correct, and it detected 59.1% of actual higher-income records. It missed 1,572 higher-income people and incorrectly classified 627 lower-income people as higher-income. The strongest global signals were marital status, relationship, education num. An individual error analysis demonstrates that population patterns can still produce the wrong prediction for a particular person.

## Risks and recommendation

Removing capital gains and losses changed F1 by -0.064; these inputs need to be available when a prediction is made. Higher-income recall differed by 9.6 percentage points between male and female records. Removing sex and race did not eliminate the need to examine group errors because other features can convey related information. Smaller race groups have less precise estimates.

Use this as an educational comparison of predictive methods. The random forest improves F1 in this evaluation, but historical data, incomplete personal information, unequal group errors and uncertain prediction-time availability limit practical use. Before any real application, define the decision and error costs, obtain suitable current data, and repeat evaluation. These results do not measure a person's ability or deserved pay. AI helped implement and document the analysis; the team should review the evidence and be able to explain the choices before submission.
