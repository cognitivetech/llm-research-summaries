# Tune without Validation Searching for Learning Rate and Weight Decay on Training Sets_chunkd2

## cognitivetech/obook_summary:q5_k_m

## Tune without Validation: Searching for Learning Rate and Weight Decay on Training Sets

"Reference: arXiv.org article (Link: [2403.05532v1](https://arxiv.org/html/2403.05532v1))"

## Abstract

**Introduction of Tune without Validation (Twin) Pipeline:**
- A new method for tuning learning rate and weight decay without requiring validation sets
- Utilizes a theoretical framework on learning phases in hypothesis space to predict optimal hyper-parameter (HP) combinations for better generalization
- **Heuristic**: Predicts best HP combinations based on early/non-early stopping scheduler, grid search of trials, and segmenting region with best training loss results
- Strong correlation between weight norm and generalization prediction within these trials
- Assessed through experiments on 20 image classification datasets across various deep network families (convolutional, transformer, feed-forward) for both scratch training and fine-tuning
- Effective in small-sample scenarios

## 1 Introduction

**Tuning Hyperparameters (HPs) without Validation Set:**
* Traditional pipelines require validation sets for hyperparameter tuning but Twin performs search directly on training sets [^26]
* Deep networks configured by a set of hyperparameters whose values impact final outcome significantly [^26]
* Incorrect configuration leads to poor performance [^35]
* Two methods for HP search: grid search and Bayesian optimization [^26]
  * Grid search: exhaustive method where all possible combinations are tested [^26]
  * Bayesian optimization: probabilistic method that uses surrogates to model the function [^26]
* Challenges in tuning with limited validation sets: noise, lack of comprehensive methodologies, and high computational cost [^35]
* Introduction of Twin: a new HP selection approach that obviates the need for validation sets [^33]
* Twin enables direct selection of LR and WD from training sets using grid search with early stopping [Fig. 1] (bottom)
* Monitors training loss as a proxy for task performance and weight norm to measure regularization strength [Sec. 3]
* Extensive empirical analysis demonstrating Twin's effectiveness across various experimental scenarios [Sec. 4.1-4.5]
  * 34 different dataset-architecture configurations [Sec. 4.1, 4.2, 4.3, 4.4]
  * Hundreds to thousands of samples, various model architectures and sizes
* Contributions: introduction of Twin for hyperparameter tuning without validation sets [Sec. 3], demonstration of its effectiveness across different experimental scenarios [Sec. 4.1-4.5].

## 2 Related Work

**Image Classification**
- Introduction of AlexNet [^26]: remarkable advances in image classification
- Development of novel neural architectures: ResNet [^16], Vision Transformer (ViT) [^11]
- Large datasets: favorable for transfer learning in small-sample scenarios [^54]
- Training deep models from scratch on limited datasets: recent work provides insights [^56]

**Hyperparameter Tuning (HP)**
- Vast literature tackling HP tuning for deep networks [^60]
  - Implicit differentiation [^35], data augmentation [^10], neural architecture search [^12], invariance learning [^55], general-purpose schedulers [^30]
- Optimization-related HPs: linear scaling rule for learning rate and batch size [^14]
  - Parameterization to zero-shot transfer LRs to larger model sizes [^58]
  - Studies on HP selection as data scales by exploiting SGD symmetries [^61]
- Few studies explore HP optimization without employing validation sets, focusing on learning invariances:
  - Bayesian inference methods fail to scale for simple tasks or larger network sizes [^43][^21]
  - Methods with strong assumptions about knowing HPs for learning invariances [^3]
  - Recent method improves scalability but requires data and model partitioning, additional backward-forward pass [^38]
- Twin focuses on LR and WD, easily scales to increased model and data sizes, simplifies HP optimization pipeline.

## 3 Tune without Validation

### 3.1 Preliminaries

**Image Classification Tasks**
- Present training set $\mathcal{D}_{train}$ and testing set $\mathcal{D}_{test}$, sampled from distribution $P(X,Y)$
- Learners (deep neural networks $f_{\theta}$) trained via SGD optimization on cross-entropy loss over training set $\min_{\theta}(\mathcal{L}_{\theta}=\mathcal{L}(f_{\theta},\mathcal{D}_{train}))$
- Regularization technique: $L_{2}$ regularization with penalty over norm of weights controlled by $\lambda$ (WD)
  - Reduces complexity of model indirectly through larger gradient noise
  - Parameters updated using momentum SGD: $\theta_{t+1}=\theta_{t}-\mu v_{t}+\alpha_{t}(\nabla\mathcal{L}_{\theta}+\lambda\theta_{t})$

**Cross-validation**
- Need a surrogate of test set for HP estimation (regularization parameter $\lambda$ and learning speed $\alpha$)
- Split training set into smaller training set $\hat{D}_{train}$ and validation set $\mathcal{D}_{val}$
  - Cardinality: $|\hat{D}_{train}|=n-m$, $|\hat{D}_{val}|=m$
- Expected error on validation set: $\mathrm{E}[\mathcal{L}(f_{\theta},\mathcal{D}_{val})]=\delta\pm\mathcal{O}(\frac{1}{\sqrt{m}})$
  - Small validation set does not provide good estimate of test error

**Motivation**
- In IID assumption, HP search is less challenging due to overlapping training and testing sets
  - Expected prediction error $\delta\approx\mathcal{L}(f_{\theta},\mathcal{D}_{train})$
- Distribution shifts (OOD learning problems) cause the dependency on validation sets for HP selection to be unreliable
- Aim to derive a robust recipe for predicting HP generalization in IID and OOD settings.

### 3.2 Working Principle

**Theoretical Framework for Learning Phases**
* Recently introduced framework explains representation learning and grokking phenomenon [^41]
* Observed in algorithmic tasks and image classification problems [^33]
* Four phases: comprehension, grokking, memorization, confusion
	+ Comprehension and grokking share goal of low training error
	+ Only these two phases reach low testing error [^33]
	+ Representation learning occurs in "Goldilocks zone" between memorization and confusion [^33]

**Intuition**
* Goal: predict generalizing configurations from hypothesis space
* Comprehension and grokking provide best generalization performance
* Low training error excludes phase with confusion (underfitting)
* Training loss can't discern overfitting from generalizing solutions
* High WD strongly penalizes parameter's norm, leading to confusion
* Low WD causes memorization due to poor regularization [Fig. 4](https://arxiv.org/html/2403.05532v1#S3.F4) supports hypothesis and shows power of parameter norm in model generalization.

**Twin's Pipeline**
* Employs gradient-based optimizer and trial scheduler for grid search across LR-WD space [Fig. 2]
* Logs train-loss and parameter-norm matrices to identify network with lowest norm within fitting region
* Parameter norm within this region is good predictor of generalization.

### 3.3 Pipeline

**Overview of Twin Algorithm**

**Twin Overview**:
- [Figure 2](https://arxiv.org/html/2403.05532v1#S3.F2 "Figure 2 ‣ Intuition. ‣ 3.2 Working Principle ‣ 3 Tune without Validation ‣ Tune without Validation: Searching for Learning Rate and Weight Decay on Training Sets") provides an overview of Twin
- Twin performs grid search over the LR-WD space and optimizes deep networks using gradient-based methods
- Trials are scheduled using a first-input-first-output (FIFO) or successive-halving-based scheduler
- A cascade of segmentation and filtering modules identifies the training loss region where generalization or overfitting occurs
- The network with the smallest parameter norm within this area is selected

**Grid Search and Trial Scheduler**:
- Twin runs $N_{	alpha}	imes N_{	lambda}$ trials sampled from a grid of equally spaced points in logarithmic space for both LR and WD
- Supports different grid densities as ablated in [Sec. 4.5](https://arxiv.org/html/2403.05532v1#S4.SS5 "4.5 Ablations ‣ 4 Experiments ‣ Tune without Validation: Searching for Learning Rate and Weight Decay on Training Sets")
- Uses a FIFO or HyperBand (HB) trial scheduler
- Default HP search over validation sets terminates a few HP configurations, while Twin needs to detect the region in the LR-WD space with low training loss

**Region Segmentation and Selection**:
- The grid search guided by the trial scheduler yields two log matrices: 1) **training losses** (matrix $\Psi$) and 2) **parameter norms** (matrix $\Theta$)
- Quickshift image segmentation algorithm is used to identify the area of the LR-WD space where the best fitting happens
- Outliers are filtered, and the values are scaled and normalized before running Quickshift segmentation
- The selected grid region is converted into a binary mask and applied to the logged $\Theta$ matrix to retrieve the final output configuration with the lowest norm

**Performance of Twin**:
- [Figure 3](https://arxiv.org/html/2403.05532v1/x3.png) shows the quantitative results, with Twin scoring an overall 1.3% MAE against the Oracle pipeline across 34 different dataset-model configurations when using a FIFO scheduler
- [Figure 4](https://arxiv.org/html/2403.05532v1/x4.png) provides a qualitative visualization of the various steps of Twin in the LR-WD space and the relationship between the selected parameter norms and test loss

## 4 Experiments

### 4.1 Overview

**Experimented Domains**
- Evaluation covers diverse domains: small datasets, medical field, natural images
- Small datasets: limited dimensions, suitable for Twin due to fewer validation sets needed
- Medical field: complexities and regulations make mitigating need for validation sets valuable
- Natural images: widely studied domain in computer vision community

**Baselines**
- **Selection from Training Set (SelTS)**: selects HP configuration with lowest loss on training set
- **Selection from Validation Set (SelVS)**: optimizes HP exclusively on validation set
  - Can be subsampled or collected externally
  - Performed with different trial schedulers: FIFO, early stopping
- **Oracle**: ideal but unrealistic scenario of selecting HPs from test set
  - Always runs a FIFO scheduler
  - Twin's best performance yields MAE of 0% vs Oracle
  - All baselines select HPs according to relevant last-epoch metric with FIFO schedulers and average of last 5 with early stopping

**Quantitative Results**
- Compare Twin against lower (SelTS) and upper (Oracle) bounds in Figure 3
- Performance per dataset with FIFO schedulers averaged across all architectures, ordered by number of images per class per dataset
- Training loss alone (SelTS) falls short as alignment between training and testing distributions decreases when IID assumption does not hold
- Twin considers regularization strength and task-learning performance for nearly optimal LR-WD configurations across all dataset scales

**Qualitative Results**
- Figure 4 presents qualitative results and empirical evidence supporting Twin's ability to predict generalizing HP configurations
- Consistency across different models (ConvNets, MLPs, transformers), optimizers (SGD, Adam), and grid densities (100 and 49 trials) demonstrated
- Training loss at last epoch, region segmentation and selection steps using Quickshift, mask application to parameter norm matrix, and relationship between parameter norm of selected sub-region and test loss shown.
- Oracle baseline's lowest loss represented by dashed green line; Twin can identify best-fitting region robustly despite various patterns and positions in LR-WD space.
- Strong relationship exists between parameter norms extracted from identified region and test loss.

### 4.2 Small Datasets

**Benchmark Datasets:**
- Five datasets spanning various domains and data types: ciFAIR-10, EuroSAT, CLaMM, ISIC 2018, CUB, Oxford Flowers
- Sub-sampled versions with 50 samples per class for ciFAIR-10, EuroSAT, CLaMM; 80 samples per class for ISIC 2018
- Image domains include RGB natural images (ciFAIR-10, CUB), multi-spectral satellite data (EuroSAT), RGB skin medical imagery (ISIC 2018), grayscale hand-written documents (CLaMM)
- Oxford Flowers comprises 102 categories with 20 images each

**Implementation Details:**
- Three model scales: ResNet-50 (RN50), EfficientNet-B0 (EN-B0), and ResNeXt-101 (32 $\times$ 8d) (RNX101) to cover tiny, small, and base classes
- Wide ResNet 16-10 (WRN-16-10) for low-resolution ciFAIR-10 images
- Perform squared grid searches of 100 trials for RN50 and EN-B0, and 36 trials for RNX101
- Set LRs and WDs intervals to $[5\cdot 10^{-5},5\cdot 10^{-1}]$ to span four orders of magnitude
- Report results with early stopping using HB ${}_{25\%}$ and ASHA schedulers for both training from scratch and transfer learning

**Results:**
*From Scratch:*
- Twin nearly matches the Oracle balanced accuracy by scoring less than 1.5% MAE across datasets and networks
- Outperforms SelVS by 71.3% vs 69.2% with EN-B0, 75.5% vs 74% with RN50, and 75.4 vs 73.7 with RNX101 when averaging performance across CUB, ISIC 2018, EuroSAT, and CLaMM
- Scalable to computationally heavy search tasks due to early stopping strategies

*With Transfer Learning:*
- The generalization of networks may differ from training from scratch depending on the source and target domain overlap
- Twin struggles in cases with strong class or feature overlap (CUB, EuroSAT RGB, ISIC 2018) when using pre-trained checkpoints, resulting in higher MAE ($\sim$ 5%)
- Employ early stopping to terminate trials whose training loss has a slower decay rate to reduce the MAE vs Oracle for CUB, EuroSAT RGB, and ISIC 2018 (< 1%)
- In case of no class or poor feature overlap (CLaMM), pre-trained checkpoints do not alter the LR-WD landscape, enabling Twin to find good HP configurations (< 2% MAE) with FIFO and HB ${}_{25\%}$ schedulers.

### 4.3 Medical Images

**Medical Images: MedMNIST v2 Benchmark**
- **Datasets**: 11 out of 12 binary/multi-class or ordinal regression tasks from the MedMNIST2D sub-collection
- **Focus**: Primary data modalities (X-ray, OCT, Ultrasound, CT, Electron Microscope) and various scales (800 to 100,000 samples)
- **Challenge**: Data diversity, maintaining computational budget
- **Testbed selection**: Pre-processed images at $28	imes 28$ resolution

**Implementation details:**
- **Model**: ResNet-18 (RN18), a modified stem for low-resolution images
- **Twin configurations**: Same as small datasets, except FIFO scheduler for both Twin and SelVS

**Results:**
- **Oracle**: Upper bound test accuracy 84.8% averaged across 11 tasks
- **Twin**: Comparable to traditional SelVS (82.9% vs 83.2%)
- **Early stopping**: Slight improvement for Twin ([Sec. 4.5](https://arxiv.org/html/2403.05532v1#S4.SS5 "4.5 Ablations ‣ 4 Experiments ‣ Tune without Validation: Searching for Learning Rate and Weight Decay on Training Sets"))
- **Cost-effectiveness**: Finds proper HPs, reduces data collection and labeling expenses (~10% of validation samples)

### 4.4 Natural Images

**Dataset Comparison:**
- **CIFAR-10/100**: CCT-2/3×2, MLP-4-512, WRN-40-2 used for testing
- **TinyImagenet**: WRN-16-4, CVT-7/8 employed
- Twin performs well on average: comparable to SelVS (71.1%) and SelTS (71.3%)
- Agnostic to data augmentation strength

**Implementation Details:**
- ConvNets: ResNet-20, ResNeXt-11, WRN-40-2
- Transformers: CCT-2/3×2
- MLP with batch normalization, ReLU activations, and hidden layers of constant width
- Data augmentation strength varies from base to strong

**Results:**
- Twin works for transformers and MLPs
- On average, comparable to SelVS (71.1%) and SelTS (71.3%), despite no validation set access
- Agnostic to data augmentation strength

**Ablation Studies:**
- Quickshift controlling segmentation density
  * MAE against Oracle baseline for different grid densities in Table 4a
- Robustness of Twin against grid density in Table 4b

**Small Datasets vs. Medical Datasets:**
- FIFO scheduler performance: EN-B0, RN50, RNX101, RN18 (Table 5)
- HB scheduler with different percentages: 25%, 12% (Table 5).

### 4.5 Ablations

**Quickshift Parameters:**
- Ablation on impact of kernel\_size and max\_dist from Quickshift
- Large max\_dist leads to poor results due to insufficient segmentation
- Setting kernel\_size and max\_dist to $\sqrt{N_{\alpha}}$ increases density for better identification

**Grid Density:**
- Robustness of Twin against different grid densities tested
- Log-spaced intervals of LR (α) and/or WD (λ) sampled every one, two, or three steps
- MAE remains almost unaffected and close to 1.3% with default settings

**Early Stopping:**
- Ablation on impact of early stopping scheduler
- Twin accommodates HB ${}_{25\%}$ or HB ${}_{12\%}$, practitioners can default to either
- Drop in performance for RNX101 with HB ${}_{12\%}$ due to small grid employed

**Optimizers and Schedulers:**
- Twin closely follows the Oracle in terms of (mean) absolute error (M)AE using various optimization setups:
  - Plain SGD, SGD with momentum (SGDM), cosine scheduler, Adam, AdamW
- Table 6 shows performance and (M)AE for different datasets, models, optimizers, and schedulers.

## 5 Conclusions

**Twin: Reliable HP Tuning Approach for Deep Networks**

- **Description**: Twin is a simple yet effective method for predicting learning rate and weight decay in deep networks without requiring validation sets.
- **Benefits**: It simplifies model selection pipelines, offers insights into the predictability of generalization for deep networks, and demonstrates robust performance across various experimental scenarios including different dataset sizes, imaging domains, architectures, scales, and training setups.
- **Scope**: This paper focuses on image classification problems; future work could investigate its application to computer vision tasks beyond images.
- **Potential Extensions**: There is potential for Twin to be extended to alternative regularization strategies beyond L2 penalty.

