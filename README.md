# Predictive and Generative AI Approaches to the Design of Deep Eutectic Solvents

This is the repository associated to the review by Luna Liviero,
Philipp Leclercq, Erwan Privat, Sergio Rampino and Antonino Polimeno
under review in the Sustainability and Circularity NOW journal.

The motivation and context for this work is described in the
aforementioned article. We focus here only on the illustrative
example results of accuracy improvement by combining the datasets
from Odegova et al.[^odegova] and Luu et al.[^luu] Available code
here is based on the first[^odegova] team code. Repository links are
in the bibliography.

## Datasets

More information on the datasets can be found on the main review
text. To summarize, the 2024 publication by Odegova et al.[^odegova]
compiled a dataset of DES properties from experimental studies dating
back to 2003, resulting in 2,303 data entries for melting
temperature, 4,369 for density, and 4,216 for viscosity. In 2023, Luu
et al.[^luu] collected a dataset with 402 entries from literature,
reporting the melting temperature as well as the component
percentages for each solvent, and successfully developed a
prompt-based transformer model for both predictive and generative
tasks, using SMILES to represent the molecules. They trained the
model jointly on DES tasks and tasks related to the well-established
and much larger QM9[^qm9] dataset.

## Our results

For illustrative purposes, and building on the data available from
recent works on DESs, computational analysis was conducted leading to
insightful exploration in regard to two aspects: the impact of
dataset size and the contribution of feature selection to model
prediction performance. The datasets compiled by Luu et al.[^luu] and
Odegova et al.[^odegova], along with the latter’s codebase, served as
the foundation for further investigation. The first step involved
compiling a unified dataset by merging the existing data frames. Luu
et al.[^luu] provides a file with 401 entries, while Odegova at
al.[^odegova] contributed 2,259 entries. The data is homogenized
through unit conversions where necessary and by retaining only the
common columns. Eight redundant entries were identified and handled
differentially. For seven of these, both data pairs were retained, as
they showed slight variations in component melting
temperatures—considered acceptable within a ±10 K margin. These
duplicates were grouped by averaging the melting temperature values
to preserve as much information as possible with some curating. After
merging, the resulting dataset contained 2,651 entries. Each DES
entry includes, where available: the SMILES formula of both
components, the melting temperature of each component, molar ratios,
the melting temperature of the resulting DES, the full names of both
components, the DES type, and the reference/DOI of the source study.
From this dataset, the final feature set and label were extracted for
model training and testing. Using the codebase from Odegova et
al.[^odegova], molecular descriptors were generated from SMILES
formulas using the RDKit library.[^rdkit] Feature selection followed
the methodology in the reference article, where over 200 descriptors
were filtered using correlation matrices to exclude highly correlated
variables. The final features included singular melting temperature,
molar fractions, molecular weight, hydrogen bond donor count, various
functional group counts, and toxicity. The target label was the
melting temperature of the DES. The models are trained on 80 % of
entries and tested on 20 % of them, using 5-fold cross-validation.
The “mixtures-out”[^muratov] method was applied to prevent data
leakage by grouping entries containing the same molecule into the
same subset. Hyperparameter optimization was performed for each
model. Evaluation metrics included the coefficient of determination
($R^2$) and root mean square error (RMSE), calculated for both
training and testing sets. The evaluation encompasses training and
testing of all models on three different variation of the datasets,
including the original two datasets. The addition of more input data
by curating and combining datasets show an overall improvement for
all models. We can note that gradient-boost-like models results were
always improved by adding more data to the base dataset. We also note
that extending the dataset also improved results while not showing
sign of overfitting for most of the tested models. This can be
explained in part by the imbalance of the data, since nearly half of
the entries belong to Type III DES and only 0.47 % to Type II. This
discrepancy is related to industrial relevance of Type III DES and
available experimental data.[^martini]

The results indicate that the relevance of the "type of DES" feature
varies across models. The kernel based models, SVM, KNN and MLP, were
the most sensitive to this feature removal with a notable drop in
performance. In contrast, having more samples always leads to
improved performance across all models. Also in this case, the kernel
models are deeply affected in their performance. This highlights the
importance of quality, structured, abundant data in the field of
machine learning to enhance model accuracy. One limitation of this
analysis lies in the uneven distribution of DES classification.
Although the additional entries lacked class information, it is
assumed that their inclusion did not attenuate the imbalance, leaving
type III and V disproportionately represented. It cannot be excluded 
that the seemingly improved model performance is driven primarily by
the overrepresented classes, while the model may struggle to capture
relationships within underrepresented DES types. Despite these
concerns, the model's performance remains significant, with three
models exceeding an $R^2$ of 0.80. The top performing model remains
Cat Boosting Regression, even after removal of a feature and with an
bigger dataset. In all conditions, it yielded the highest average
$R^2$ on the test subset and as this value rose sharply, the
corresponding training increased mildly. This suggests that the model
performance improved on unseen data points and has therefore good
generalization capabilities. SVM and KNN also demonstrated notable
potential, and may outperform CBR in future applications if the DES
type feature is reintroduced into the augmented dataset.

In the table below is a recap of the cross-validation metrics $R^2$
and RMSE for original and extended dataset for several models. These
results are from targeting melting temperature and RMSEs are in
Kelvin.


| Model | Original $R^2$ | Extended $R^2$ | Original RMSE | Extended RMSE|
|-------|----------------|----------------|---------------|--------------|
| DTR   | 0.583633       | 0.652694       | 47.3243       | 47.7102      |
| RFR   | 0.713469       | 0.785704       | 39.3110       | 37.2780      |
| GBR   | 0.733936       | 0.788667       | 37.9431       | 36.6524      |
| CBR   | 0.757018       | 0.823381       | 36.2145       | 33.8333      |
| XGB   | 0.733484       | 0.803071       | 38.0284       | 35.6826      |
| SVM   | 0.629013       | 0.812928       | 43.7242       | 34.8591      |
| KNN   | 0.642345       | 0.805520       | 43.0502       | 35.3635      |
| MLP   | 0.469012       | 0.771632       | 52.9626       | 38.4747      |

**Regression models:**

* DTR: decision tree
* RFR: random forest
* GBR: gradient boosting
* CBR: cat boosting
* XGB: extreme boosting
* SVM: support vector machine
* KNN: K-nearest neighbors
* MLP: multilayer perceptron


[^odegova]: Odegova, V.; Lavrinenko, A.; Rakhmanov, T.; Sysuev, G.;
    Dmitrenko, A.; Vinogradov, V. Green Chem., 2024, 26, 3958.
    https://github.com/lamm-mit/MoleculeDiffusionTransformer

[^luu]: Luu, R.K.; Wysokowski, M.; Buehler, M.J. Appl. Phys. Lett.
    2023, 122, 234103.
    https://github.com/lamm-mit/MoleculeDiffusionTransformer

[^qm9]: Junde Li, Swaroop Ghosh (2024). Dataset: QM9.
    https://doi.org/10.57702/exp0m45r

[^rdkit]: RDKit: Open-source cheminformatics. https://www.rdkit.org

[^muratov]: Muratov, E. N., Varlamova, E. V., Artemenko, A. G.,
    Polishchuk, P. G., & Kuz'min, V. E. (2012). Existing and
    developing approaches for QSAR analysis of mixtures. Molecular
    informatics, 31(3‐4), 202-221.

[^martini]: Petteri Vainikka, Sebastian Thallmair, Paulo Cesar Telles
    Souza, and Siewert J. Marrink, ACS Sustainable Chemistry &
    Engineering 2021 9 (51), 17338-17350, DOI:
    10.1021/acssuschemeng.1c06521

[^lundberg]: Lundberg, S. M.; Lee, S.-I. A Adv. neural inf. process.
    syst. 2017, 30, arXiv.1705.07874, DOI:
    https://doi.org/10.48550/arXiv.1705.07874.

[^rezende]: Rezende, D.; Mohamed, S. In Proceedings of the 32nd
    International Conference on Machine Learning; PMLR, 2015, 1530.
