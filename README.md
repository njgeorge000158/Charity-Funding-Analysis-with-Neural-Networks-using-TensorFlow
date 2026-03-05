<img width="769" alt="alphabet_charity_1" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/dde52c9f-5beb-418f-af31-6bb0b4386971">

----

# **Alphabet Soup Funding Success Predictor: A Deep Learning Binary Classification Model**

---

## **Project Overview**

Alphabet Soup, a philanthropic organization, faces a challenge common to all grant-making institutions: how to allocate limited funding to the applicants most likely to use it successfully. To address this, I developed a deep learning binary classification model trained on a historical dataset of over 34,000 organizations that have previously received funding. The model's objective is to predict, with maximum accuracy, whether a future applicant will use their funding successfully — equipping Alphabet Soup with a data-driven screening tool to improve the impact of every dollar granted.

---

## **Data Preprocessing**

![charity_analysisTable25UpdatedCharityDataTable](https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/dafdc9fa-c8ce-4bcb-8a37-9b85a14effb8)

The dataset, represented in Table 2.5, captures a rich set of organizational attributes for each applicant, including application type, affiliation, classification, intended use case, organization type, income level, and funding amount requested. Before modeling could begin, several preprocessing decisions were made to maximize data quality and model performance.

The target variable for the binary classification task is `IS_SUCCESSFUL` — a binary flag indicating whether the funded organization used its grant effectively. Eight variables were retained as model features: `NAME`, `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `INCOME_AMT`, and `ASK_AMT`. Three columns were excluded entirely: `EIN`, `STATUS`, and `SPECIAL_CONSIDERATIONS`. These fields were dropped because they contributed no predictive signal — `EIN` is a unique identifier with no generalizable pattern, while `STATUS` and `SPECIAL_CONSIDERATIONS` exhibited near-zero variance, meaning virtually all records shared the same value and the fields could not meaningfully differentiate outcomes.

To manage the high cardinality of several categorical features and reduce the distorting influence of outliers, I applied binning — grouping rare categories under an `OTHER` label using carefully calculated cutoff thresholds for each feature variable: NAME (1), APPLICATION_TYPE (156), AFFILIATION (33), CLASSIFICATION (6,074), USE_CASE (146), ORGANIZATION (0), INCOME_AMT (728), and ASK_AMT (53). This technique of converting sparse numerical distributions into consolidated categorical bins proved highly effective at reducing noise without sacrificing meaningful signal — one of the key methodological insights of this project.

---

## **Model Architecture**

<img width="657" alt="Screenshot 2024-05-07 at 9 45 07 AM" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/e94b2c58-5c17-44a2-9c16-2292760e6311">

As shown in the Model Summary in Image 2, the final optimized model is a sequential neural network comprising three layers: two hidden layers and one output layer, with a total of 48,548 trainable parameters across 189.64 KB of model weight.

The first hidden layer contains 57 neurons and accounts for the vast majority of the model's parameters — 47,367 — as it receives the full width of the encoded input feature space. A dropout layer follows immediately after, serving as a regularization mechanism to reduce overfitting by randomly deactivating neurons during training. The second hidden layer narrows to 20 neurons, adding a further level of abstraction with 1,160 parameters, followed by a second dropout layer for continued regularization. The output layer consists of a single neuron — appropriate for binary classification — contributing just 21 parameters.

Both hidden layers use SeLU (Scaled Exponential Linear Unit) activation functions, which are well-suited to deep networks as they enable self-normalizing behavior, helping to maintain stable gradient flow through the network during training. The output layer uses a Linear activation function, paired with a Huber loss function — a robust choice that is less sensitive to outliers than mean squared error while retaining the smoothness of squared loss near zero. The model is compiled with the Adamax optimizer, a variant of the Adam optimizer that performs reliably in the presence of sparse gradients and noisy feature spaces, both of which are characteristic of this dataset.

This architecture was deliberately designed to balance expressive capacity against the risk of overfitting — a key challenge when working with tabular data of moderate dimensionality. The progressive narrowing from 57 to 20 to 1 neuron creates a funnel structure that forces the network to compress its learned representations at each stage, encouraging generalization rather than memorization.

---

## **Model Evaluation and Results**

<img width="925" alt="Screenshot 2024-05-07 at 9 46 47 AM" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/078081b0-ea0e-4e5b-bebf-063e5433c12a">

As shown in Image 3, the optimized model achieved a predictive accuracy of **81.41%** with a model loss of **6.78%** on the held-out test set — evaluated across all 215 test batches in under one second. These results comfortably exceed the project's target accuracy threshold of 75%, representing a meaningful improvement of more than six percentage points over the benchmark.

The low loss value of 6.78% indicates that the model's probability estimates are well-calibrated — it is not merely classifying correctly but doing so with appropriate confidence. Together, these metrics suggest a model that has learned genuinely predictive patterns from the training data rather than overfitting to noise.

---

## **Summary and Future Directions**

This project demonstrates that a carefully designed and optimized sequential neural network can serve as an effective and practical funding screening tool for a charitable organization. The most impactful methodological contributions were the strategic elimination of low-signal columns, the application of cutoff-based binning to manage cardinality and outlier effects, and the architectural choices — particularly the SeLU activation functions, dropout regularization, and Huber loss — that collectively enabled the model to generalize well beyond the training data.

Looking ahead, several avenues exist for further performance improvement. The current implementation is constrained to sequential neural network architectures. Expanding the optimization search to include alternative configurations — such as residual connections, attention mechanisms, or ensemble approaches combining neural networks with gradient-boosted trees — could push accuracy beyond the current ceiling. Additionally, more aggressive feature engineering, particularly around the `ASK_AMT` and `INCOME_AMT` variables where outlier distributions were most pronounced, may yield further gains. Finally, exploring alternative optimizers and loss functions — or implementing learning rate scheduling — could improve convergence behavior and reduce loss further.

Ultimately, a model that can predict funding success with over 81% accuracy from application data alone represents a meaningful and deployable tool — one that could help Alphabet Soup direct its resources more effectively, maximize its charitable impact, and make better-informed decisions at every stage of the grant selection process.

----

### Copyright

Nicholas J. George © 2023. All Rights Reserved.
