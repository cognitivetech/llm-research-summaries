# I want to break some laws too

source: https://snats.xyz/pages/articles/breaking_some_laws.html
by Santiago Pedroza

### TLDR:

I automated a pipeline to clean data, starting from the Minipile paper. It led me down a rabbit hole: careful data selection breaks scaling laws, and being a "data snob" pays off.

## Contents
- [Introduction](#introduction)
- [Minipile](#minipile)
	- [Deciding on a dataset](#deciding-on-a-dataset)
	- [The elbow method](#the-elbow-method)
	- [Few shot prompting for cluster classification](#few-shot-prompting-for-cluster-classification)
- [Examples](#examples)
	- [Example 1](#example-1)
- [Cluster to Classify](#cluster-to-classify)
	- [Something, something, success!](#something-something-success)
- [Going from Power to Exponential](#going-from-power-to-exponential)
	- [All roads lead to Physics](#all-roads-lead-to-physics)
	- [What data do we go for?](#what-data-do-we-go-for)
	- [Building the last step](#building-the-last-step)
- [Conclusion](#conclusion)
	- [Takeaways](#takeaways)
- [Appendix:](#appendix)
	- [Making Datacomp run](#making-datacomp-run)

## Introduction

**Impact of Datasets on AI Models**
* **Factors influencing performance:** Neural network architecture, training method, data used for training
* Importance of clean, ready-to-use datasets: faster experimentation and learning
* Pile dataset pruning example: 0.72% size with comparable results to original
	+ BERT: drop of only 1.9% in GLUE and SNI benchmarks
	+ T5: drop of 2.5% in GLUE and SNI benchmarks
* Inspiration for research: pruning datasets, comparing results with original versions

**Pile Dataset Pruning Methodology**
* Reduced size from original to 0.72%
* Comparable results despite significant data reduction

**Questions Arising from Research:**
* How does the dataset pruning algorithm work?
* Applicability of this approach to other dataset styles?
* Potential limits for dataset pruning technique?

## Minipile

**Comparison of Text Datasets:** Size vs. Diversity and Content Composition

**The Pile Dataset**:
- Small compared to modern datasets (825 GB)
- Significant at the time, used by many researchers
- Compared to: RedPajama-V2 (270 TB), Dolma v1.7 (4.5 TB)

**The Minipile Dataset**:
- Toy dataset, only 6 GB in size
- Contains 1 million samples from the original data source
- Smaller but still provides good training data for small language models

**Cleaning Pipeline for The Pile Dataset**:
1. Generate embeddings for the entire dataset
2. Cluster embeddings using k-mean clustering
3. **Manually** discard low quality clusters

**Criticisms of The Pile Dataset Cleaning Method**:
- Manual work is not ideal, can be automated
- Choosing the number of clusters was subjective (10 per subset)
- Other objective methods like elbow or silhouette method could be used instead.

**Implementation**:
- Need to implement the cleaning pipeline using a language model and few-shot prompting.

### Deciding on a dataset

**Choosing a Dataset:**
- **DataComp Homepage**: Discovered recently for training CLIP and text model
- Competition focuses on accuracy by improving dataset, not code
- Several track sizes from Small (12.8 million images) to XLarge (12.8 billion)
- Documented with a paper and baselines provided
- Embeddings already generated for all images and captions

**Downloading the Dataset:**
- Small track: 528 GBs, downloaded locally
- Images used on separate VM
- Ignored DNS resolvers and link rot issues, downloaded roughly 80% of the original dataset.

**Full Pipeline for Minipile:**
- Training CLIP and text model using DataComp's Small track dataset
- Diagram available: https://snats.xyz/assets/breaking_the_law/minipile_pipeline.svg

### The elbow method

**Elbow Method for Determining Optimal Number of Clusters**
- **Vibes-based approach**: not as accurate for determining number of correct clusters
- **Elbow method**: empirical but better than pure vibes
- **Inertia**: measure of how packed the clusters are, lower inertia = better fit to data
- Elbow plot: visual representation of improvement in results with increasing clusters, plateaus after a certain point
- Running k-means with different numbers of clusters and calculating inertia for each run

**Results from Text Embeddings Clustering**
- Cluster 60 example: semantically similar but not well-defined (bad)
- Cluster 25 example: more distinct, some unrelated samples included (not so bad)
- Impressed by the results, especially considering only text embeddings were used

**Next Steps**
- Use best cluster number to run k-means on all data
- Label each image and caption with its corresponding group for further analysis.

### Few shot prompting for cluster classification

**Cluster Quality Classifier for CLIP Model Training**

**Purpose**:
- Evaluate image-text clusters to determine "High Value" or "Low Value" for training high-quality CLIP models
- Identify clusters with diverse, informative, and high-quality samples that benefit CLIP's understanding of visual concepts from natural language supervision
- Avoid clusters containing repetitive, low-quality, or potentially harmful content

**Approach**:
1. Use an AI model for labeling clusters instead of manual methods
2. Provide examples and guidelines to the model
3. Utilize Few Shot Prompting with Chain of Thought reasoning
4. Consider two different prompts: [prompt 1](https://github.com/snat-s/m/blob/main/luvdatacomp/prompt.txt) and [prompt 2](https://github.com/snat-s/m/blob/main/luvdatacomp/only_nearest_points_prompt.txt)
5. Use the five nearest examples to each cluster for evaluation
6. Train CLIP models using high-quality clusters
7. Learn visual concepts from natural language supervision
8. Improve understanding and connection between images and text.

## Examples

### Example 1

Cluster data: "closest_samples": [ "Mountain landscape", "Rare flower", "European architecture", "Indigenous portrait", "Microscopic cell" ]

Reason: Diverse, informative content with high-quality imagery and educational value. Good for training CLIP on various subjects and scales. Classification: High Value

## Cluster to Classify

**Cluster Data Analysis**

**DeepSeek Usage**:
- Used DeepSeek due to prompt caching by default
- Reduced classification costs: mostly cache hits
- Examples of model responses:

**Cluster 1 - Brand Focused (Low Value)**:
- Repetitive content on "Givenchy" brand
- Limited diversity in subjects
- No clear educational or informative value
- Classification: Low Value

**Cluster 2 - Diverse Cultural and Educational Materials (High Value)**:
- Variety of literary, historical content
- Study guides, music recordings, fantasy books, horror eBooks, etc.
- Rich set of text-image pairs for CLIP to understand different types of cultural materials
- Classification: High Value

**Post-Processing**:
- Clean up responses using regexes.

### Something, something, success!

**Model Training Results:**
- **Models trained**: my\_baseline, minipile\_style\_trained\_img\_txt, minipile\_style\_trained\_txt, txt\_top5\_all\_quality\_clusters, txt\_top5\_english\_quality\_clusters
- **Baseline from DataComp paper**: 0.025 (ImageNet), 0.033 (dist. shifts), 0.145 (VTAB), 0.114 (Retrieval), 0.132 (Average)
- **my\_baseline**: 0.026, 0.034, 0.148, 0.112, 0.137, 12,800,000, 10,386,623, **81.15%**
- **minipile\_style\_only\_txt**: 0.010, 0.018, 0.134, 0.067, 0.111, 933,381, 739,116, **5.77%**
- **minipile\_style\_txt\_img**: 0.021, 0.025, 0.120, 0.077, 0.114, 1,633,210, 1,290,236, **10.08%**
- **txt\_top5\_all\_quality\_clusters**: 0.022, 0.031, 0.132, 0.102, 0.126, 3,660,046, 2,864,016, **22.38%**
- **txt\_top5\_english\_quality\_clusters**: 0.015, 0.026, 0.145, 0.081, 0.121, 1,712,451, 1,316,522, **10.29%**

**Performance Comparison:**
- Baseline from DataComp was slightly worse than my\_baseline
- Using only top 5 examples nearest to centroid performed well: drop of .005 on ImageNet, good results in Retrieval (0.081) and Labels Seen
- English model performed well in Retrieval with a drop of 0.003
- Trend line shows that more data leads to better performance, but my\_baseline outperformed the baseline from DataComp paper.

## Going from Power to Exponential

**Minipile Paper Insights:**
* Second paragraph caught attention due to clustering embeddings ([Minipile et al., 2021](https://arxiv.org/abs/2206.14486))
	+ Motivated by recent work on clusterability of data subset embeddings [25, 54]
* Reference to "Probabilistic Active Meta-Learning" ([Li et al., 2020](https://arxiv.org/abs/2007.08949)) sounded intimidating
* Opted for second reference: "Beyond neural scaling laws: beating power law scaling via data pruning" ([Liu et al., 2022](https://arxiv.org/abs/2206.14486))
	+ Introduce idea of breaking scaling laws
		- Power laws describe how models improve with more data or larger sizes, often following a power law relationship
	+ Showed empirical and theoretical proof for their approach
	+ Claim: "you could go from power law scaling to exponential scaling"
* Skimming other paper revealed similarities to method used. (Minipile et al.)
	+ Main idea: generate embeddings, grab a subset of examples from each centroid.

### All roads lead to Physics

This paper uses the replica method from statistical mechanics to analyze complex systems. In this case, our dataset is a complex system of particles (data points) with two sources of randomness: underlying data distribution and sample choice. By applying the replica method, we can predict average ML model performance across datasets. However, we also need to determine which training data will yield better performance, allowing us to break power laws and achieve exponential improvement.

### What data do we go for?

**Data Selection Strategies for Machine Learning Models:**

**Decision Tree for Data Selection:**
- If data is abundant: focus on hard examples
- If data is limited: focus on easy examples
- Gradient between the two scenarios
- Optimal data selection method proposed by authors:
  - Hardest or easiest examples without human supervision
  - Similar to Minipile's approach for code reuse

**Pareto Optimal Frontier:**
- Identifies the sweet spot where best performance is achieved with least amount of data

**Data Refinement and Model Training Effort:**
- The more you refine a dataset, the less compute required for model training
- Amortizing cost of training through refined datasets: Foundational dataset
- Observation from research paper: fewer examples needed to train models effectively [1]

**Data Pruning Techniques:**
- Minipile magic and neural scaling law to be further optimized [2]
- Example of FineWeb-Edu: impressive results with a fraction of the entire dataset [3]

**References:**
[1] Paper performance models graph (source: paper)
[2] Snats.xyz/assets/breaking_the_law/paper_performance_models.png
[3] https://HuggingFace.Co/spaces/HuggingFaceFW/blogpost-fineweb-v1
[4] Minipile magic and neural scaling law techniques not perfectly effective but main idea remains true [5] Footnote 5 in the original text.

### Building the last step

**Pipeline for Reducing Dataset Size**:
- **Generate embeddings for your dataset**
- **Do KMeans clustering on them**
- If you have few examples, use nearest centroid points; if many examples, use furthest away from centroid

**Experiment Goals**:
1. See how much the dataset can be pruned to get comparable results to the original baseline
2. Test scaling capabilities at home

**Dataset Pruning**:
- Pruned 90% of data up to the baseline (hard examples)
- Changed only dataset size, kept same hyperparameters and compute

**Loss Chart Findings**:
1. Model trained with 10% data overfits more than others
2. Other models' final cross entropies closer to each other

**Accuracy Chart Findings**:
- Accuracy increases up to 80%, then decreases a little
- "Supervised" methods performed better but lacked images

**Log Scaling Findings**:
- Logarithmic plot shows trend of fast increase, tapering off with larger datasets

**Conclusion**:
- Results not as good as original DataComp baselines, but learned a lot from the experiment.

## Conclusion

**Discussion on Pipeline Building**
* Liking progress from scratch to creation of functional pipeline (supervised or unsupervised)
* Improvements for Minipile: use both text and images
* Statistical method only works with images, Minipile for text
* Both methods effective on various data styles

**Expansion of Research**
* Exploring larger scales in second method to check for saturation point
* Open source code available in mono repo [here](https://github.com/snat-s/m/tree/main/luvdatacomp)
* Dataset training lists on Hugging Face [here](https://huggingface.co/datasets/snats/datacomp_lists)
* Models from different runs in repo [here](https://huggingface.co/snats/clip-datacomp-models)
* Access to downloaded version of DataComp small [here](https://huggingface.co/datasets/snats/small_datacomp_all) with images included.

### Takeaways





* More data doesn't always lead to better results.
* Pruning data breaks neural scaling laws, changing from power laws to exponential scaling.
* Refining datasets reduces the amount of training required.
* We should focus on creating foundational datasets that reduce the cost of AI training over time.

## Appendix:

### Making Datacomp run

**Datacomp Environment Setup**
- **Dependencies:** Install `build-essential`, `libgl1-mesa-glx` using apt:
  - `apt install build-essential libgl1-mesa-glx -y`
- **Modify environment.yml**: Change version of `pyyaml` to `6.0`
  - Find the line that starts with `- pyyaml=`, and change it to `- pyyaml=6.0`
- **Activate Conda Environment:**
  - Source the conda environment file: `source /opt/conda/etc/profile.d/conda.sh`
  - Activate the Datacomp environment: `conda activate datacomp`

**Notes**:
- Many research repositories are not maintained and have dependency issues.
- To run Datacomp in a fresh vast.ai machine, follow these steps.
- Install necessary packages using apt.
- Modify the `environment.yml` file to change the version of `pyyaml`.
- Activate the Conda environment.

