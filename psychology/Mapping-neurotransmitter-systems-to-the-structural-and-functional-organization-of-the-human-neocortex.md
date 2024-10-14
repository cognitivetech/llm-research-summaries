# Mapping neurotransmitter systems to the structural and functional organization of the human neocortex (2022)

by Justine Y. Hansen, Golia Shafiei, Ross D. Markello;_Nature Neuroscience_

https://github.com/netneurolab/hansen_receptors

https://www.nature.com/articles/s41593-022-01186-3

## Contents
- [Abstract](#abstract)
- [Main](#main)
- [Results](#results)
  - [Receptor distributions reflect structural and functional organization](#receptor-distributions-reflect-structural-and-functional-organization)
  - [Receptor profiles shape oscillatory neural dynamics](#receptor-profiles-shape-oscillatory-neural-dynamics)
  - [Mapping receptors to cognitive function](#mapping-receptors-to-cognitive-function)
  - [Mapping receptors and transporters to disease vulnerability](#mapping-receptors-and-transporters-to-disease-vulnerability)
  - [Replication using autoradiography](#replication-using-autoradiography)
  - [Sensitivity and robustness analyses](#sensitivity-and-robustness-analyses)
- [Discussion](#discussion)
- [Comprehensive Brain Receptor Mapping in Health and Disease](#comprehensive-brain-receptor-mapping-in-health-and-disease)
- [Methods](#methods)
  - [PET data acquisition](#pet-data-acquisition)
  - [Autoradiography receptor data acquisition](#autoradiography-receptor-data-acquisition)
  - [Structural and functional data acquisition](#structural-and-functional-data-acquisition)
  - [Structural network reconstruction](#structural-network-reconstruction)
  - [Functional network reconstruction](#functional-network-reconstruction)
  - [Structure–function coupling](#structurefunction-coupling)
  - [MEG power](#meg-power)
  - [ENIGMA cortical abnormality maps](#enigma-cortical-abnormality-maps)
  - [Dominance analysis](#dominance-analysis)
  - [Cognitive meta-analytic activation](#cognitive-meta-analytic-activation)
  - [Partial least squares analysis](#partial-least-squares-analysis)
  - [Distance-dependent cross-validation](#distance-dependent-cross-validation)
  - [Null models](#null-models)
  - [Reporting Summary](#reporting-summary)
- [Data availability](#data-availability)
- [Supplementary information](#supplementary-information)

## Abstract

**Neurotransmitter Receptors:**
- Support signal propagation in human brain
- Location and impact on emergent function not well understood
- No comprehensive atlas of receptors available

**Collating Positron Emission Tomography (PET) Data:**
- Constructed a whole-brain three-dimensional normative atlas of 19 receptors and transporters across nine neurotransmitter systems
- Data from over 1,200 healthy individuals used

**Findings:**
- Receptor profiles align with structural connectivity
- Mediate function, including neurophysiological oscillatory dynamics and resting-state hemodynamic functional connectivity

**Topographic Gradient of Overlapping Receptor Distributions:**
- Separates extrinsic and intrinsic psychological processes using Neurosynth cognitive atlas

**Associations between Receptor Distributions and Cortical Abnormality Patterns:**
- Found both expected and novel associations in an independently collected autoradiography dataset

**Implications:**
- Demonstrates how **chemoarchitecture** shapes brain structure and function
- Provides a new direction for studying multi-scale brain organization.

## Main

**Neurotransmitter Receptors and Brain Function**

**Concepts**:
- **Neurotransmitter receptors**: Heterogeneously distributed across neocortex, modulate excitability and firing rate of cells, mediate transfer and propagation of electrical impulses, drive synaptic plasticity, shape network communication.
- **Ionotropic vs. metabotropic receptors**: Ionotropic affect membrane potential directly; metabotropic interact with intracellular second messengers.
- **Neurotransmitter systems**: Segregated and integrated information through specialized modules and hubs, respectively.

**Challenges in Studying Neurotransmitter Receptors**:
- Lack of comprehensive datasets for receptor distributions across multiple neurotransmitter systems.
- Autoradiography data available only in 44 cytoarchitectonically defined cortical areas; PET data limited by small cohorts and high cost.

**Data Sharing Efforts**:
- Comprehensive atlas of neurotransmitter receptor maps from 19 unique receptors, binding sites, and transporters across 9 neurotransmitter systems and over 1,200 healthy individuals: [https://github.com/netneurolab/hansen\_receptors](https://github.com/netneurolab/hansen_receptors).

**Integrating Neurotransmitter Receptor Data with Other Modalities**:
- Diffusion-weighted MRI and functional MRI: Neurotransmitter receptor densities follow brain's structural and functional connectomes.
- MEG: Neurotransmitter receptor densities shape oscillatory neural dynamics.
- Neurosynth functional activations: Spatially co-varying axis of neuromodulators and mood-related processes.
- ENIGMA cortical atrophy patterns: Specific receptor–disorder links.

## Results

**Neurotransmitter Receptor Profiles**

**PET Images**:
- Collated PET images from 19 different neurotransmitter receptors, transporters, and receptor-binding sites across 9 neurotransmitter systems
- Included dopamine, norepinephrine, serotonin, acetylcholine, glutamate, GABA, histamine, cannabinoid, and opioid systems
- Acquired in healthy participants

**Data Preprocessing**:
- Parcellated PET tracer maps into the same 100 cortical regions
- Z-scored the data to mitigate acquisition and pre-processing variations

**Analysis**:
- Presented tracer maps for 19 unique neurotransmitter receptors and transporters from a combined total of 1,238 healthy participants
- Repeated analyses in an independently collected autoradiography dataset of 15 neurotransmitter receptors
- Examined across alternative brain parcellations

**Results**:
- Obtained mean receptor distribution maps for 19 different neurotransmitter receptors and transporters, totaling over 1,200 healthy participants
- Included non-displaceable binding potential (BPND), tracer distribution volume (VT), density (Bmax), standard uptake value ratio (SUVR) values

**Note**:
- Neurotransmitter receptor maps without citations refer to previously unpublished data
- Contact information for study principal investigators is provided in Supplementary Table [3](https://www.nature.com/articles/s41593-022-01186-3#MOESM3)
- Table 3 includes methodological details such as PET camera, number of males and females, modeling method, reference region, scan length, and modeling notes

### Receptor distributions reflect structural and functional organization

**Constructing a Cortical Neurotransmitter Receptor Atlas**

**Receptor Similarity**:
- Quantified by correlating receptor density profiles of brain regions
- Decreases exponentially with Euclidean distance between brain region centroids
- Approximately normally distributed
- No single receptor or transporter exerts undue influence

**Principal Component of Receptor Density**:
- Represents a regional quantification of receptor similarity
- Separates insular and cingulate from somatomotor/posterior parietal regions
- Correlated with synapse density, supporting the notion that receptor expression depends on lamination

**Stratifying Receptors**:
- By biological mechanisms (excitatory/inhibitory, ionotropic/metabotropic, Gs-/Gi-/Gq-coupled)
- By neurotransmitter protein structure (monoamine/non-monoamine)

**Relationship to Structural and Functional Connectivity**:
- Receptor similarity is greater between anatomically connected brain regions
- Not due to spatial proximity or network topography, as shown by significance against surrogate structural connectivity matrices

**Correlation of Receptor Similarity with Structural and Functional Connectivity:**
- Significant correlation between receptor similarity and structural connectivity (_P_ = 1.6 × 10−8, CI = [0.11, 0.23], two-sided) (Fig. 3a)
  * Receptor similarity greater between physically connected regions
  * Regression analysis shows positive correlation with structural connectivity
- Significant correlation between receptor similarity and functional connectivity (_P_ = 7.1 × 10−61, CI = [0.20, 0.26], two-sided) (Fig. 3b)
  * Receptor similarity greater within same functional networks
  * Positive correlation with functional connectivity

**Receptor Profiles and Structure–Function Coupling:**
- Communicability of weighted structural connectome used to measure structure–function coupling
- Inclusion of receptor similarity improves prediction of regional functional connectivity in unimodal areas and paracentral lobule (Fig. 3c)
- Significance assessed against null distribution using a distance-dependent method.

### Receptor profiles shape oscillatory neural dynamics

**Neurotransmitter Receptors and Neural Oscillations**

**Relating Cortical Patterning of Neurotransmitter Receptors to Neural Oscillations:**
- Analyzed MEG power spectra across six canonical frequency bands from HCP[29](#ref-CR29 "Van Essen, D. C. et al. The Wu-Minn Human Connectome Project: an overview. Neuroimage 80, 62–79 (2013). ")[30](#ref-CR30 "Shafiei, G., Baillet, S. & Misic, B. Human electromagnetic and haemodynamic networks systematically converge in unimodal cortex and diverge in transmodal cortex. PLoS Biol. 20, e3001735 (2022).")
- Fit multiple linear regression models that predict cortical power distribution of each frequency band from neurotransmitter receptor and transporter densities
- Cross-validated the model using a distance-dependent method
- Assessed significance against a spin-permuted null model (10,000 repetitions) and found all models except high-gamma are significant after FDR correction (_P_spin < 0.05, one-sided)

**Receptor Densities and MEG-Derived Power:**
- Close fit between receptor densities and MEG-derived power distributions
- Suggests overlapping spatial topographies of multiple neurotransmitter systems manifest as coherent oscillatory patterns

**Dominance Analysis:**
- Applied to identify independent variables contributing most to the fit
- Dominance analysis assigns proportion of final fit to each input variable for statistically significant models
- Found that MOR (opioid), H3 (histamine), and α4β2 make large contributions to lower-frequency (theta and alpha) as well as low-gamma power bands[31](#ref-CR31 "Azen, R. & Budescu, D. V. The dominance analysis approach for comparing predictors in multiple regression. Psychol. Methods 8, 129–148 (2003).")
- Prominence of ionotropic receptors in autoradiography dataset replication[32](#ref-CR33 "Witjes, B. et al. Magnetoencephalography reveals increased slow-to-fast alpha power ratios in patients with chronic pain. Pain Rep. 6, e928 (2021).")
- Inhibitory, non-monoamine and Gi-coupled receptors more dominant than excitatory, monoamine and Gs-/Gq-coupled receptors, respectively[31](#ref-CR31 "Azen, R. & Budescu, D. V. The dominance analysis approach for comparing predictors in multiple regression. Psychol. Methods 8, 129–148 (2003).") (Supplementary Fig. [5a](https://www.nature.com/articles/s41593-022-01186-3#MOESM1))

### Mapping receptors to cognitive function

**Brain Mapping: Receptors and Cognitive Function**
* **Neurosynth meta-analytic task activation maps**: derived from multiple cognitive tasks to identify brain regions activated during various functions
* **Partial least squares (PLS) analysis**: used to identify the relationship between neurotransmitter receptors/transporters and functional activation maps
* Significant latent variable (_P_spin = 0.010, one-tailed) representing 54% of covariance between receptor distributions and Neurosynth-derived cognitive functional activation
* **Receptor scores** and **cognitive scores**: reflect how well a brain area exhibits the dominant spatial pattern of receptor distributions and cognitive activations, respectively
* Receptor and cognitive score patterns reveal a sensory-fugal spatial gradient separating limbic, paralimbic, insular cortices from visual and somatosensory cortices
* Cross-validation using distance-dependent method (_mean out-of-sample Pearson's r_(98) = 0.54, _P_spin = 0.046, one-sided)) demonstrates a link between receptor distributions and cognitive specialization

**Receptor loadings**: correlation (Pearson's _r_) between each receptor's distribution across the cortex and PLS-derived scores; contribution of each receptor to the latent variable
* Almost all receptors/transporters have positive loading, with metabotropic dopaminergic and serotonergic receptors having the greatest loadings
* Cognitive processes with large positive loadings are enriched for emotional and affective processes such as 'emotion', 'fear', and 'valence'
* NET (norepinephrine transporter) has stable negative loading, co-varies with functions such as 'fixation', 'planning', and 'skill' in primarily unimodal regions.

**Implications**: These results demonstrate a direct link between cortex-wide molecular receptor distributions and functional specialization.

### Mapping receptors and transporters to disease vulnerability

**Neurotransmitter Receptors and Transporters in Diseases and Disorders**

**Identifying Neurotransmitter Receptors/Transporters**:
- Important for developing new therapeutic drugs based on specific disorders
- Relating neurotransmitter receptors/transporters to cortical abnormality patterns across neurological, developmental, and psychiatric disorders

**Methods**:
- Used datasets from the ENIGMA consortium for 13 disorders:
    - 22q11.2 deletion syndrome
    - Attention deficit hyperactivity disorder (ADHD)
    - Autism spectrum disorder (ASD)
    - Idiopathic generalized epilepsy (IGE)
    - Right and left temporal lobe epilepsy
    - Depression
    - Obsessive-compulsive disorder (OCD)
    - Schizophrenia
    - Bipolar disorder (BD)
    - Obesity
    - Schizotypy
    - Parkinson's disease (PD)
- Fit a multiple regression model to predict each disorder's cortical abnormality pattern from receptor and transporter distributions
- Assessed the significance of each model against an FDR-corrected one-sided spatial autocorrelation-preserving null model
- Evaluated each model using distance-dependent cross-validation

**Results**:
- **Receptor Distributions**:
    - Some disorders are more heavily influenced by receptor distribution than others
    - IGE and schizotypy show low and non-significant correspondence with receptor distributions
    - ADHD, autism, and temporal lobe epilepsies show greater correspondence with receptor distributions
- **Serotonin Transporter (5-HTT) Distributions**:
    - Contribute more to OCD, schizophrenia, and BD profiles than any other receptors
- **Mu-Opioid Receptor**:
    - Strongest contributor to ADHD cortical abnormality patterns
    - Consistent with findings from animal models
- **Unexpected Relationships**:
    - In PD, dopamine receptors are not implicated
    - Serotonin receptors do not make large contributions to depression

**Conclusion**:
- These results present an initial step towards a comprehensive "look-up table" that relates neurotransmitter systems to multiple brain disorders.

### Replication using autoradiography

**Neurotransmitter Receptor Densities in the Brain**

**PET Imaging vs Autoradiography**:
- PET imaging provides estimates for neurotransmitter receptor densities, but:
  - Densities are acquired from PET imaging alone
  - Quantification methods vary across radioligands, image acquisition protocols, and preprocessing
- Autoradiography is an alternative technique to measure receptor density:
  - Captures local densities at a defined number of postmortem brain sections
  - High cost and labor intensity limit the availability of a complete 3D autoradiography cross-cortex atlas

**Authoradiography Dataset**:
- Includes 15 neurotransmitter receptors (8 not included in PET dataset)
- Consists of ionotropic and metabotropic receptors, including excitatory glutamate, acetylcholine, and norepinephrine receptors

**Similarity Between PET and Authoradiography**:
- **Receptor similarity** is significantly correlated between the two datasets:
  - Pearson's _r_(1033) = 0.38, _P_ = 6.7 × 10−38, CI = [0.33, 0.44]
- **Receptor gradients** are also correlated:
  - Pearson's _r_(44) = 0.51, _P_perm = 0.0001, CI = [0.26, 0.70], two-sided

**Authoradiography Receptor Densities**:
- Follow similar architectural patterns as PET-derived receptor densities:
  - **Receptor similarity** is non-significantly greater between structurally connected brain regions (_P_ = 0.19)
  - Significantly correlated with structural connectivity (Pearson's _r_(329) = 0.39, _P_ = 1.4 × 10−13, CI = [0.30, 0.48])
  - Greater in regions within the same intrinsic network (_P_spin = 0.03)
  - Significantly correlated with functional connectivity (Pearson's _r_(1033) = 0.21, _P_ = 1.1 × 10−12, CI = [0.16, 0.28])
- Augment structure–function coupling in visual, paracentral, and somatomotor regions

**Authoradiography Receptor Densities vs MEG Oscillations**:
- AMPA, NMDA, GABAA, and α4β2 (all ionotropic receptors) are most dominant in fitting autoradiography neurotransmitter receptors to MEG power
- Confirms that fast oscillatory dynamics captured by MEG are closely related to fluctuations in neural activity modulated by ionotropic neurotransmitter receptors

**Authoradiography Receptor Densities vs Cognitive Functional Activation and Disease Vulnerability**:
- Authoradiography-derived receptor densities follow similar topographic gradients linking to Neurosynth-derived functional activations
- PET-derived and autoradiography-derived receptor and cognitive scores are correlated
- Loadings of receptors and cognitive processes are consistent
- Prominent associations between authoradiography-derived receptor densities and cortical abnormality patterns of multiple disorders, including a relationship between the ionotropic glutamate receptor kainate and depression

### Sensitivity and robustness analyses

**Findings:**
- **Methodological robustness**: analyses repeated using different parcellation resolutions and receptor subsets confirm consistency of results
- **Single neurotransmitter's influence**: no single neurotransmitter receptor/transporter disproportionately influences receptor similarity, as evidenced by highly correlated original and iteratively removed receptor similarity matrices
- **Age effects**: age has negligible effect on reported findings, as shown by high correlation between age-regressed and original receptor density and similarity matrices. However, individual subject variability in neurotransmitter systems may not be captured by mean age analysis.

**Methods:**
- **Parcellation resolution**: results consistent using parcellations of 200 and 400 cortical regions
- **Single receptor/transporter removal**: highly correlated original and iteratively removed receptor similarity matrices confirm no disproportionate influence of a single neurotransmitter receptor/transporter on receptor similarity
- **Age effects analysis**: linear model fit between mean age of scanned participants and receptor density, resulting in age-regressed receptor density and similarity matrices with high correlation to original.

## Discussion

**Neurotransmitter Receptors and Brain Organization**

**Key Findings**:
- Comprehensive 3D atlas of 19 neurotransmitter receptors and transporters
- Chemoarchitecture is a key layer in the multi-scale organization of the brain
- Neurotransmitter receptor profiles align with structural connectivity and mediate function, including neurophysiological oscillatory dynamics and resting-state hemodynamic functional connectivity
- Overlapping topographic distributions of receptors manifest as patterns of cognitive specialization and disease vulnerability

**Background**:
- Brain's structural architecture gives rise to its function
- Connectomics model represents brain's structural or functional architectures as regional nodes interconnected by links, with the assumption of homogenous nodes
- Emerging effort to annotate connectome with molecular, cellular, laminar attributes

**Neurotransmitter Receptors and Transporters**:
- Important molecular annotation for bridging brain structure to function
- Previous initiatives used autoradiography to map receptor densities in human and macaque brains
- Consistent results between autoradiography and PET datasets

**Receptor Distribution and Brain Structure/Function**:
- Prominent link between receptor distribution and brain structure and function
- Canonical electrophysiological frequency bands can be captured by overlapping topographies of multiple receptors
- Multivariate mapping between receptor profiles and cognitive activations

**Implications**:
- Serotonergic and dopaminergic receptors underlying patterns of cognitive activation related to affect
- Robust spatial concordance between multiple receptor maps and cortical abnormality profiles across brain disorders
- Key step toward developing therapies for specific syndromes is to reliably map them onto underlying neural systems

## Comprehensive Brain Receptor Mapping in Health and Disease

**Neurotransmitter Receptor Profiles and Disease Phenotypes**

**Background:**
- Study findings on neurotransmitter receptors and their associations with disease phenotypes
- Some results have preliminary support in literature but not clinically adopted
  - Histamine H3 in PD: Rinne et al. (2002)
  - MOR in ADHD: Sagvolden et al. (2009)
  - D1 and NET in TLE: Costa et al. (2016), Giorgi et al. (2004)

**Implications:**
- Mapping disease phenotypes to receptor profiles can identify novel targets for pharmacotherapy
- Present report focuses on cortical thinning/thickening but should be expanded in future work
- Building on previous neurochemical and pharmacological causal studies
- Large-scale characterization of receptor systems should be validated and inspire future research

**Potential Avenues for Future Research:**
- Study changes in receptor architecture in healthy aging, across sexes, and mapping to subcortical structures
  - Dopamine D1 and D2: Seaman et al. (2019)
  - Serotonin transporter and receptor density: Karrer et al. (2019)
  - GABAergic differences: Cuypers et al. (2021)
- Greater understanding of multi-system receptor distributions across age, sex, and within subcortical structures

**Methodological Considerations:**
- Use of PET images with low spatial resolution and no laminar specificity
  - Replicated using autoradiography dataset, but comprehensive atlas needed for full understanding
- Normalized spatial distributions to focus on relative topographies
- Accounted for spatial dependencies using autocorrelation-preserving null models
- Restricted analyses to cortex, obscuring subcortical neuromodulatory systems
- Direct comparison between PET and autoradiography datasets not possible due to missing receptors in the PET datasets.

## Methods

**Data and Code Availability**:
- Code and data for the analyses are accessible at [github.com/netneurolab/hansen_receptors](https://github.com/netneurolab/hansen_receptors)
- Volumetric PET images can be found in neuromaps at [github.com/netneurolab/neuromaps](https://github.com/netneurolab/neuromaps), where they can be easily converted between template spaces (Markello et al., 2022, Nature Methods)

### PET data acquisition

**PET Images and Neurotransmitter Receptors/Transporters:**
* Volumetric PET images collected for 19 different neurotransmitter receptors and transporters across multiple studies (n = 1,238; 718 males, 520 females)
* Protect patient confidentiality by averaging individual participant maps within studies before sharing
* Details: Table [1](https://www.nature.com/articles/s41593-022-01186-3#Tab1), Supplementary Table [3](https://www.nature.com/articles/s41593-022-01186-3#MOESM3)
* Only healthy participants were scanned using best practice imaging protocols recommended for each radioligand (ref.[56](https://www.nature.com/articles/s41593-022-01186-3#ref-CR56))
* Images registered to MNI-ICBM 152 template and parcellated into 100, 200, or 400 regions according to Schaefer atlas (ref.[12](https://www.nature.com/articles/s41593-022-01186-3#ref-CR12))
* Receptors/transporters with more than one mean image of the same tracer combined using weighted average (Supplementary Fig.[13a](https://www.nature.com/articles/s41593-022-01186-3#MOESM1))
* Each tracer map corresponding to each receptor/transporter z-scored and concatenated into a final region by receptor matrix of relative densities
* Comparisons between tracers shown for some neurotransmitter receptors/transporters: 5-HT1A, 5-HT1B, 5-HT2A, 5-HTT, CB1, D2, DAT, GABAA, MOR, and NET (Supplementary Fig.[13b](https://www.nature.com/articles/s41593-022-01186-3#MOESM1))
* Specific notes: 5-HTT and GABAA involve comparisons between the same tracers (DASB and flumazenil, respectively), but one map is converted to density using autoradiography data (ref.[9](https://www.nature.com/articles/s41593-022-01186-3#ref-CR9))

**Serotonin System Mapping**: High-resolution in vivo atlas of human brain's serotonin system using multiple tracers: [FLB457](#ref-CR8), [raclopride](#ref-CR98), and [fallypride](#ref-CR103).
* FLB457 is a reliable SERT tracer for mapping serotonin densities in the cortex.
* Raclopride has unreliable binding in the cortex, but its comparison to FLB457 and fallypride is presented for completeness.

**Dopamine Receptor Mapping**: Carfentanil (MOR) map collated from PET Turku Centre database due to overlap with an alternative map; no combination of tracers into a single mean map.
* Importance of choosing appropriate tracers for accurate mapping.

**Synaptic Density Measurement**: [11C]UCB-J, a PET tracer that binds to the synaptic vesicle glycoprotein 2A (SV2A), used to measure synapse density in 76 healthy adults.
* Data collected using HRRT PET camera for 90 minutes after injection and modeled using SRTM2 with centrum semiovale as reference.
* Group-averaged map presented in ref. [105].

### Autoradiography receptor data acquisition

**Receptor Autoradiography Data**
* Originally acquired as described in [Zilles & Palomero-Gallagher (2017)](#ref-CR6)
* Fifteen neurotransmitter receptor densities across 44 cytoarchitectonically defined areas in three postmortem brains (age range: 72–77 years, two males)

**Receptor Densities Included**:
- See [Supplementary Table 1](https://www.nature.com/articles/s41593-022-01186-3#MOESM1) for a complete list
- See Supplementary Table 2 in [Zilles & Palomero-Gallagher (2017)](#ref-CR6) for originally reported densities
- Access machine-readable Python numpy files at [https://github.com/AlGoulas/receptor_principles](https://github.com/AlGoulas/receptor_principles)

**Comparison to PET Data Analyses**:
* Manually created region-to-region mapping between the 44 autoradiography cortical areas and the 50 left hemisphere Schaefer regions
* Four regions in Schaefer atlas did not have a suitable mapping to the autoradiography atlas, resulting in conversion to 46 Schaefer left hemisphere regions
* Concatenated and z-scored receptor densities to create a single map of receptor densities across the cortex.

### Structural and functional data acquisition

**Data Acquisition and Pre-Processing**

**Participants**:
- 326 unrelated participants (age range: 22–35 years) from the HCP S900 release
- 145 males

**Functional MRI Data**:
- Two scans on day 1 and two scans on day 2
- Each scan approximately 15 minutes long
- **TR** = 720 ms

**Diffusion-Weighted Imaging (DWI) Data**:
- Available for all participants

**Pre-Processing**:
- All structural and functional MRI data pre-processed using HCP minimal pre-processing pipelines
- Detailed information on data acquisition and pre-processing available elsewhere

### Structural network reconstruction

**Preprocessing DWI Data using MRtrix3**
* Fiber orientation distributions generated using multi-shell, multi-tissue constrained spherical deconvolution algorithm from MRtrix:
  * Estimated response function without co-registered T1 image (Dhollander et al., 2016; Jeurissen et al., 2014)
* White matter edges reconstructed using probabilistic streamline tractography based on generated fiber orientation distributions:
  * Improved accuracy with 2nd order integration (Tournier, Calamante & Connelly, 2010)
* Tract weights optimized by estimating appropriate cross-section multiplier for each streamline:
  * SIFT2 procedure (Smith et al., 2015)
* Connectivity matrix built for each participant using 100-region Schaefer parcellation:
  * Local-global parcellation of human cerebral cortex (Schaefer et al., 2018)

**Group Consensus Binary Network Construction**
* Preserve density and edge-length distributions of individual connectomes:
  * Distance-dependent consensus thresholds (Betzel et al., 2019)
* Assign weights to edges in group consensus network:
  * Average log-transformed streamline count across participants
* Scale edge weights between 0 and 1.

### Functional network reconstruction

**Preprocessing Steps for Functional MRI Data:**
* Correction of gradient non-linearity, head motion, and geometric distortions using scan pairs with opposite phase encoding directions[106](#ref-CR106 "de Wael, R. V. et al. Anatomical and microstructural determinants of hippocampal subfield functional connectome embedding. Proc. Natl Acad. Sci. USA 115, 10154–10159 (2018).")
* Co-registration of corrected images to T1w structural MR images
* Brain extraction and normalization of whole brain intensity
* High-pass filtering (>2,000s full width at half maximum (FWHM)) to correct for scanner drifts
* Removing additional noise using the ICA-FIX process[106](#ref-CR106 "de Wael, R. V. et al. Anatomical and microstructural determinants of hippocampal subfield functional connectome embedding. Proc. Natl Acad. Sci. USA 115, 10154–10159 (2018). "),[114](#ref-CR114 "Salimi-Khorshidi, G. et al. Automatic denoising of functional MRI data: combining independent component analysis and hierarchical fusion of classifiers. Neuroimage 90, 449–468 (2014).")
* Parcellation of pre-processed time-series into 100 cortical brain regions according to the Schaefer atlas[12](#ref-CR12 "Schaefer, A. et al. Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. Cereb. Cortex 28, 3095–3114 (2018).")
* Construction of functional connectivity matrices as Pearson correlation coefficient between pairs of regional time series for each scan
* Group-average functional connectivity matrix construction by taking the mean of all individual and scan functional connectivity matrices.

### Structure–function coupling

**Brain Connectivity Metrics:**
* **Communicability**: measure of diffusion between brain regions based on weighted average of all walks and paths between them.
* Calculated as: simple linear regression model fits regional communicability to functional connectivity.
* Represents diffusive communication in complex networks (Crofts & Higham, 2009; Estrada & Hatano, 2008).
* Bridges structure and function (Seguin et al., 2022).

**Structure–function Coupling:**
* Defined as the adjusted _R_2 of a simple linear regression model that fits:
  * Regional communicability to regional functional connectivity.
* Significance assessed against null distribution of adjusted _R_2 from a model with rotated receptor similarity vector (10,000 repetitions).
* Ensures robustness of increased _R_2 when receptor information is included in the model.

**Receptor Similarity:**
* Included as an additional independent variable in the structure–function coupling model.
* Represents similarity between a region of interest and every other region.
* Significance assessed against null distribution to ensure robustness of results.

### MEG power

**MEG Data Acquisition and Preprocessing**
- **Six-minute resting-state MEG time series**: acquired from HCP (S1200 release) for 33 unrelated participants
- Participant demographics: age range 22–35 years, 17 males
- **Complete MEG acquisition protocols** can be found in HCP S1200 Release Manual

**MEG Data Processing**
- Power spectrum computed at vertex level across six frequency bands: delta (2–4 Hz), theta (5–7 Hz), alpha (8–12 Hz), beta (15–29 Hz), low gamma (30–59 Hz), high gamma (60–90 Hz) using Brainstorm
- **Pre-processing**:
  - Apply notch filters at 60, 120, 180, 240 and 300 Hz to remove artifacts
  - High-pass filter at 0.3 Hz to remove slow-wave and DC offset artifacts
- **Source estimation**:
  - Preprocessed sensor-level data used to obtain source activity on HCP's fsLR4k cortex surface for each participant
  - Head models computed using overlapping spheres, and data/noise covariance matrices estimated from MEG and noise recordings
  - Brainstorm's LCMV beamformers method applied to obtain source activity
- **Power spectrum density (PSD) estimation**:
  - Welch's method used with overlapping windows of length 4 seconds and 50% overlap
- **Source-level power data parcellation**: divided into 100 cortical regions for each frequency band

### ENIGMA cortical abnormality maps

**ENIGMA (Enhancing Neuroimaging Genetics through Meta-Analysis) Consortium:**
* Data-sharing initiative for standardized image acquisition and processing pipelines
* Disorder maps from ENIGMA consortium and Enigma toolbox: [https://github.com/MICA-MNI/ENIGMA](https://github.com/MICA-MNI/ENIGMA) (ref. [118](#ref-CR118))
* Patterns of cortical abnormality collected for various disorders: 22q11.2 deletion syndrome, ADHD, ASD, idiopathic generalized epilepsy (right temporal lobe, left temporal lobe), depression, OCD, schizophrenia, BD, obesity, schizotypy, and PD.

**Disorders and Cortical Abnormalities:**
- **22q11.2 deletion syndrome**: Large-scale mapping of cortical alterations (ref. [119](#ref-CR119))
- **ADHD**: Brain imaging of the cortex in ADHD (ref. [120])
- **ASD**: Cortical and subcortical brain morphometry differences between patients with ASD and healthy individuals (ref. [121])
- **Idiopathic generalized epilepsy**: Structural brain abnormalities in the common epilepsies (ref. [122])
- **Depression**: Cortical abnormalities in adults and adolescents with major depression (ref. [123])
- **OCD**: Cortical abnormalities associated with pediatric and adult OCD (ref. [124])
- **Schizophrenia**: Cortical brain abnormalities in schizophrenia patients (ref. [125])
- **BD**: Cortical abnormalities in BD patients (ref. [126])
- **Obesity**: Brain structural abnormalities in obesity (ref. [127])
- **Schizotypy**: Cortical and subcortical neuroanatomical signatures of schizotypy (ref. [128])
- **PD**: International multicenter analysis of brain structure across clinical stages of PD (ref. [129])

**Cortical Abnormalities:**
* Decreases in cortical thickness: most disorders
* Increases in cortical thickness: 22q, ASD, and schizotypy

**Data Collection:**
* Over 21,000 scanned patients and almost 26,000 controls
* Adult patients (except for ASD) following identical processing protocols
* Values are z-scored effect sizes (Cohen’s _d_) of cortical thickness in patient populations versus healthy controls
* Native representation: Desikan–Killiany atlas (68 cortical regions) (ref. [130])

**Imaging and Processing Protocols:**
[http://enigma.ini.usc.edu/protocols/]

### Dominance analysis

**Dominance Analysis**
- **Purpose**: Determine relative contribution of independent variables to overall fit (adjusted _R\_2) of multiple linear regression model
- **Methodology**:
  - Fit the same regression model on every combination of input variables (2\_p\_ - 1 submodels for a model with _p_ input variables)
  - Calculate total dominance: average of the relative increase in _R\_2 when adding a single input variable to a submodel, across all combinations
  - Sum of dominance equals total adjusted _R\_2 of complete model
  - Normalize by total fit (_R\_2) for comparability within and across models
- **Advantages**:
  - Accounts for predictor–predictor interactions
  - Interpretable
  - Partitions the total effect size across predictors

### Cognitive meta-analytic activation

**Neurosynth: Probabilistic Measures of Association between Voxels and Cognitive Processes**

**Background:**
- Neurosynth is a meta-analytic tool synthesizing results from 15,000 published functional MRI studies
- Focuses on high-frequency keywords like 'pain' and 'attention' with voxel coordinates
- Provides probabilistic measures of association between voxels and cognitive processes

**Data Collection:**
- Searches for cognitive and behavioral terms in Neurosynth (Cognitive Atlas)
  - Umbrella terms: attention, emotion
  - Specific cognitive processes: visual attention, episodic memory
  - Behaviors: eating, sleeping
  - Emotional states: fear, anxiety
- Terms selected from Cognitive Atlas, a public ontology of cognitive science
- Coordinates reported by Neurosynth are parcellated according to the Schaefer-100 atlas and z-scored

**Interpretation:**
- Probabilistic measure reported by Neurosynth: quantitative representation of how regional fluctuations in activity relate to psychological processes
- Full list of cognitive processes is available in Supplementary Table [2](https://www.nature.com/articles/s41593-022-01186-3#MOESM1).

### Partial least squares analysis

**Neurotransmitter Receptor Distributions and Functional Activation Analysis using Partial Least Squares (PLS)**

**Technique Overview**:
- Unsupervised multivariate statistical technique: PLS
- Decomposes two datasets into orthogonal sets of latent variables with maximum covariance

**Latent Variables**:
- **Receptor weights**, **cognitive weights** and a singular value representing the covariance between receptor distributions and functional activations

**Scoring Process**:
1. Project original data onto respective weights
2. Assign brain regions receptor and cognitive scores
3. Compute **receptor loadings**: Pearson's correlation between receptor densities and receptor scores
4. Compute **cognitive loadings**: Pearson's correlation between functional activations and cognitive scores

**Significance Analysis**:
- Assess significance of latent variable based on singular value against spin-test
- Only the first significant latent variable was analyzed further
- Cross-validate empirical correlation between receptor and cognitive scores using distance-dependent validation

**Limitations**:
- Does not establish causal relationships between receptors and cognition
- Does not make specific univariate associations between receptors and cognitive function
- Does not preclude existence of additional relationships between receptors and cognitive function.

### Distance-dependent cross-validation

**Assessment of Multilinear Models**

**Robustness Assessment**:
- Cross-validated using a **distance-dependent method** to assess the robustness of each multilinear model
- Applied to:
    - Every multilinear regression model (Figs. 3c, 4 and 6)
    - The PLS model (Fig. 5)
- Procedure:
    - Selected the **75% closest regions** as the training set
    - Remaining **25% of brain regions** as the test set
    - Stratification procedure to minimize dependence due to spatial autocorrelation

**Multilinear Regression Models**:
- Model fit on the training set
- Predicted test set output variable (regional functional connectivity, MEG power or disorder maps) was correlated to the empirical test set values
- Distribution of **Pearson's correlations** between predicted and empirical variables across all repetitions found in Supplementary Figs. 2, 3 and 7

**PLS Analysis**:
- Model fit on the training set
- Weights were projected onto the test set to calculate predicted receptor and cognitive scores
- Training and test sets defined as described above
- Correlation between receptor and cognitive score was calculated in both the training and test set
- Significance of mean out-of-sample correlation assessed against a permuted null model (Fig. 5d)

### Null models

**Spatial Autocorrelation-Preserving Permutation Tests**

**Assessing Statistical Significance of Associations Across Brain Regions**:
- Termed "spin tests"
- Used to assess statistical significance of associations across brain regions
- Designed by Alexander-Bloch et al. (2018)[24](#ref-CR24 "Alexander-F. et al., On testing for spatial correspondence between maps of human brain structure and function, Neuroimage 178, 540–551 (2018).")
- Implemented by Markello and Misic (2021)[25](#ref-CR25 "Markello, R. D. & Misic, B., Comparing spatial null models for brain maps, Neuroimage 236, 118052 (2021).")

**Creating a Surface-Based Representation**:
- Defined spatial coordinates for each parcel on the FreeSurfer fsaverage surface
- Rotated parcel coordinates and reassigned parcels to nearest rotated parcel (10,000 repetitions)
- Handled medial wall cases separately

**Performing Spin Test**:
- Not applied to autradiography data
- Permutation test used instead

**Testing Receptor Similarity in Connected vs. Unconnected Regions**:
- Generates a null structural connectome preserving density, edge length, and degree distributions
- Swaps edges within distance bins to create rewired networks
- Computes the difference between mean receptor similarity of unconnected and connected edges
- Compares this difference to a null distribution computed on the rewired networks

### Reporting Summary

**Research Design Information**

Find further details in the [Nature Research Reporting Summary](https://www.nature.com/articles/s41593-022-01186-3#MOESM2) accompanying this article.

## Data availability

**Data Availability:**
- **Volumetric PET images**, including receptor images and synaptic density: [github.com/netneurolab/hansen\_receptors](https://github.com/netneurolab/hansen_receptors)
- Neuromaps conversion between template spaces: [github.com/netneurolab/neuromaps](https://github.com/netneurolab/neuromaps)
- **Autoradiography data**: Supplementary Table 2 of ref. [6](#ref-CR6 "Zilles, K. & Palomero-Gallagher, N. Multiple transmitter receptors in regions and layers of the human cerebral cortex. Front. Neuroanat. 11, 78 (2017).")
- **HCP dataset**: [db.humanconnectome.org/](https://db.humanconnectome.org/)
- Neurosynth data: [neurosynth.org/](https://neurosynth.org/)
- ENIGMA datasets: ENIGMA consortium and the ENIGMA Toolbox ([github.com/MICA-MNI/ENIGMA](https://github.com/MICA-MNI/ENIGMA))
- **Parcellation atlases**: netneurotools ([github.com/netneurolab/netneurotools](https://github.com/netneurolab/netneurotools))
  - Schaefer-100 and Desikan–Killiany atlas

## Supplementary information

**Supplementary Materials**
- Figures 1–13 and Tables 1 and 2 can be found at [this link](https://static-content.springer.com/esm/art%3A10.1038%2Fs41593-022-01186-3/MediaObjects/41593_2022_1186_MOESM1_ESM.pdf)
- Reporting Summary and Supplementary Table 3 are available at [this link](https://static-content.springer.com/esm/art%3A10.1038%2Fs41593-022-01186-3/MediaObjects/41593_2022_1186_MOESM2_ESM.pdf)
- [Excel file](https://static-content.springer.com/esm/art%3A10.1038%2Fs41593-022-01186-3/MediaObjects/41593_2022_1186_MOESM3_ESM.xlsx) contains methodological details for each PET tracer.

