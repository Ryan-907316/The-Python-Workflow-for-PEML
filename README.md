# The Python Workflow for Physics Enhanced Machine Learning

(Currently untitled, but in future versions a name will be given)

Code developed by Ryan Cheung as part of a 3rd Year Individual Project at Lancaster University

Title: Developing a Python workflow to aid in Physics-Enhanced Machine Learning applications in engineering

Date: Friday, March 28th 2025

Python Version used: 3.12.6

Dependencies: See requirements.txt

Abstract:

Engineering systems often exhibit complex nonlinearities due to factors such as friction, temperature, and material properties. Traditional physics-based simulations, which rely on solving complex equations to model real-world systems- such as airflow over a plane wing or heat transfer in materials- are highly accurate, but computationally expensive. Machine Learning (ML) offers a faster alternative by learning patterns from data, yet it often struggles to make accurate predictions on new, unseen cases. Physics-Enhanced Machine Learning (PEML) combines scientific equations with ML, improving prediction accuracy while ensuring models adhere to key principles of real-world physics, such as energy conservation. To evaluate its practical effectiveness, this project applies PEML to an engineering problem: modelling a DC motor’s speed and torque, where nonlinearities challenge both traditional simulations and standalone ML models.
To achieve this, a Python-based workflow was developed to systematically implement and evaluate PEML for DC motor modelling. The workflow includes data preprocessing, feature engineering, hyperparameter optimisation, interpretability analysis, and statistical validation to ensure rigorous model evaluation. Standalone ML and PEML models were evaluated using Leave-One-Group-Out (LOGO) validation to test generalisation, and a Kruskal-Wallis statistical test assessed whether PEML’s improvements over ML were statistically significant. The workflow was tested on DC motor datasets synthesised in MATLAB, comparing PEML against standalone ML. Results showed that PEML significantly outperformed standalone ML in predicting motor speed, achieving a predictive R^2 of 0.93- a 55.7% improvement. Additionally, PEML increased cross-validation scores by 19% and reduced the two-standard deviation prediction spread by 67%, improving robustness. However, for motor torque, PEML’s improvements were marginal or even detrimental, likely because the governing equations were already sufficiently accurate.
These findings highlight that PEML is highly effective when physical models fail to capture nonlinearities but may be unnecessary when governing equations already provide strong estimates. The workflow offers a structured approach for determining when and where PEML is most beneficial in engineering applications. Future work could explore applying this workflow to other engineering applications or the addition of features in the workflow, such as dimensionality reduction, multi-objective optimisation, or real-time ML integration.

The workflow includes:

- Box and scatter plots of features and target variables
- Distance correlation matrix for feature selection
- Model training and evaluation of multiple regression algorithms using scikit-learn
- Bootstrapping and conformal prediction uncertainty quantification
- PDP, ICE and SHAP plots for model interpretability
- Hyperparameter optimisation methods such as Random Search, Hyperopt, and Scikit-Optimize
- Model comparison and result exporting using Pickle
- Postprocessing and Cross-Validation
- Cook's Distance plots and influential point identification
- Histogram of residuals, automatic transformation of residuals, and Q-Q plots

To run the workflow, follow these steps:
1)	Install Python for your operating system. This workflow was written using Python version 3.12.6 and is guaranteed to work in this version. Other versions may also work, and any version of Python from 3.8 and above should be acceptable.
Visit https://www.python.org/downloads/ to download Python for your system and follow the instructions to download Python. Ensure the “Add Python to PATH” option is ticked during installation.
2)	Download the .py file and place it in a known directory on your system. 
3)	Open the requirements.txt and copy all the names of the packages. Then, open a terminal and run:
>>pip install [PASTE PACKAGES HERE]

Alternatively, the user may also download the requirements.txt file and run this command to run install everything at once:

>>pip install -r requirements.txt

4)	After all the packages have been installed, navigate to the workflow .py file, and run the program. The workflow should begin without any issues.

If the reader encounters any issues running the script, please contact the author via this email: cheungkh@lancaster.ac.uk

This report uses Version 0.1.0 of the workflow. Future versions may have updated dependencies, so ensure that the appropriate version of the workflow and packages are used.

Note: This workflow uses pywin32, a library that only works on Windows operating systems. As of version 0.1.0, this workflow is only compatible with Windows.

To customise the workflow to work on your device:

On line 367, copy the dataset's path from your computer that you wish to analyse. Note that this dataset has to be appropriate, currently this workflow only supports numerical regression and will not accept datasets with empty or NaN values.

On line 675-700, you are free to add or remove any models not included in the workflow. However, ensure omitted models are commented out in lines 689-700 and models that are added are defined and properly embedded. The default hyperparameters can be changed on lines 482-534 and the hyperparameter search space can be changed on lines 540-594.

On line 1873, specify the path on your computer to where you want your hyperparameter optimisation results to be stored.

On line 2539, if you wish to load your .pkl file (or any .pkl file), paste the path here.

In terms of workflow setting customisation:


Test/train split: Line 364

Display a dummy for the distance correlation matrix: Line 460

Uncertainty Quantification settings: Lines 852-858

ICE, PDP and SHAP plot settings: Lines 1047-1050

Random Search settings: Lines 1212 to 1216

Hyperopt settings: Lines 1429 to 1430

Scikit-optimize settings: Lines 1610 to 1612

Cross Validation settings: Lines 2069-2081
