# Fully automatic extraction of morphological traits from the Web: utopia or reality?

by Diego Marcos, Robert van de Vlasakker, Ioannis N. Athanasiadis, Pierre Bonnet, Hervé Goeau, Alexis Joly, W. Daniel Kissling, César Leblanc, André S.J. van Proosdij, Konstantinos P. Panousis

https://arxiv.org/pdf/2409.17179

## Contents
- [Abstract](#abstract)
- [INTRODUCTION](#introduction)
- [METHODS](#methods)
  - [Species-trait datasets for evaluation](#species-trait-datasets-for-evaluation)
  - [Textual data harvesting and description detection](#textual-data-harvesting-and-description-detection)
  - [Trait information extraction.](#trait-information-extraction)
  - [Evaluation metrics](#evaluation-metrics)
- [RESULTS](#results)
  - [Descriptive text harvesting](#descriptive-text-harvesting)
  - [Automatic trait extraction](#automatic-trait-extraction)
  - [Evaluation of the false negative rate](#evaluation-of-the-false-negative-rate)
  - [Additional experimental results](#additional-experimental-results)
- [DISCUSSION](#discussion)
- [Appendix](#appendix)

## Abstract
**Premise**:
- Plant morphological traits are crucial for understanding species' roles within their ecosystems
- Compiling trait information for a large number of species is a labor-intensive task requiring expert knowledge and extensive time
- Massive amounts of information about plant descriptions are available online in unstructured text form, making it difficult to utilize at scale

**Method**:
- Propose to use recent advances in Large Language Models (LLMs) to gather and process information on plant traits from unstructured textual descriptions without manual curation

**Results**:
- Evaluated the approach by automatically replicating three manually created species trait matrices
- Achieved values for over half of all species-trait pairs with an F1-score above 75%

**Discussion**:
- The results suggest that large-scale creation of structured trait databases from unstructured online text is currently feasible due to the information extraction capabilities of LLMs, limited only by the availability of textual descriptions covering all the traits of interest

**Keywords**:
- Automatic trait extraction
- Large language models
- Morphological trait matrices
- Natural language processing

## INTRODUCTION
- Traits are observable characteristics used to answer questions about ecology, evolution, and usefulness to humans
- Morphological traits are main cues for species identification since taxonomy's advent
- Complexity of trait descriptions makes it challenging to create comprehensive frameworks across taxonomic groups
- Recent efforts advocate for standard vocabulary and open science initiatives to make trait databases compatible [Schneider et al., 2019; Gallagher et al., 2020]

**Existing Databases:**
- Limited in terms of geographic or taxonomic scope
- Largest community efforts like TRY, BIEN, TraitBank not comprehensive or representative [Kattge et al., 2020]
- Imbalance between species with many traits vs. those having fewer than 10 [Kattge et al., 2020]
- Taxonomists describe traits for identification since taxonomy's dawn, now used with machine learning approaches [Almeida et al., 2020]
- Expertise and large amounts of data available as textual descriptions but require curation [Endara et al., 2018; Folk et al., 2023]

**Potential Solution:**
- Leveraging recent advances in NLP models, particularly Large Language Models (LLMs), to exploit textual knowledge automatically
- LLMs behave as zero-shot learners, capable of solving tasks without training examples via natural language instructions [Kojima et al., 2023]
- Excel at extracting structured information from text [Wei et al., 2023]

**Proposed Workflow:**
- Given names of species and traits/values, fill in a species-trait matrix using web crawling and LLMs
- Differentiates from other related approaches requiring manual post-processing or preparing training sets [Endara et al., 2018; Folk et al., 2023; Coleman et al., 2023; Domazetoski et al., 2023]

## METHODS

**Proposed Framework**
- Requires three inputs:
  - List of species of interest
  - List of traits of interest
    - For each trait, list of allowed values
- Output: Species-trait table indicating which trait values pertain to each species

**Workflow Steps**
1. **Textual Data Harvesting**:
   - Search engine API used to retrieve URLs relevant to the species name
   - Text content is downloaded from these URLs
2. **Description Detection**:
   - Binary classification NLP model filters out irrelevant text
3. **Trait Information Extraction**:
   - Language model (LLM) detects all possible categorical trait values within descriptive text

### Species-trait datasets for evaluation

**Species-Trait Datasets for Evaluation**

**Caribbean**:
- 42 woody species in the Dutch Caribbean
- Contains 24 traits, with an average of 8.5 possible values per trait (minimum of 2 and a maximum of 22)

**West Africa [Bonnet et al., 2005]**:
- 361 species of trees in the West African savanna
- Consider all 23 traits, averaging 5.8 possible values per trait (minimum of 2 and a maximum of 10)

**Automatic Trait Extraction Workflow**:
1. **Data Harvesting**:
   - Fixed to three manually created species-trait matrices
2. **Description Detection**:
   - Identify "Species" and "Trait" sections in data
3. **Trait Extraction**:
   - Parse trait names and possible values from text

**Outputs**:
- Search engine
- Sentence list (10 lines)

**Example Input**:
- Wikipedia URL for Hedera helix
  - Includes species name and GBIF ID
- Leaf arrangement example: "Alternate, Opposite, Whorled"

**Palms Dataset**:
- 333 species with complete trait description
- Six categorical traits
  - Average of 9.5 possible values per trait (minimum of 2 and a maximum of 31)

### Textual data harvesting and description detection

**Textual Data Harvesting**
- Use Google Search API to retrieve HTML sites containing specific species names
- Scrape text from first 20 returned URLs after confirming the presence of the species name in the HTML header
- Select sentences most likely to be part of a morphological description using a custom text classifier

**Description Detection**
- Develop an approach for distinguishing between descriptive and non-descriptive sentences
- Leverage structured online sources (Wikipedia, Plant of the World Online, Encyclopedias of Living Forms, World Agroforestry Center) with labeled sections to create a training dataset
  - Descriptive text: "Appearance" and "Characteristics" sections
  - Non-descriptive text: "Introduction" and "Habitat" sections
- Train a description detector using DistillBERT model with added classification head and soft bootstrap consistency objective for robustness against noisy labels.

### Trait information extraction. 
#### Information extraction with a generative LLM

**Information Extraction with a Generative LLM+**

**Trait Information Extraction**:
- Involves extracting relevant information from obtained text snippets into a structured form
- Leverages recent advances in Language Models (LLMs) to capture relational knowledge
- LLMs perform well on generic tasks like common sense or general knowledge [Ouyang et al., 2022; Davison et al., 2019; Petroni et al., 2019]
- However, LLMs tend to provide unfounded "hallucinations" in specialized domains like botany due to long-tail distributions

**Mitigating Hallucination Issues**:
- Turn the task into **information extraction from text via search engine retrieval**
- Provide LLM with descriptive text and questions about a predetermined set of traits
- Allow LLM to explicitly state if requested information is "Not Available (NA)"
- This ensures only a subset of traits are assigned values, referred to as the **coverage rate**

**Choice of LLM**:
- The 32k-token context window in **mistral-medium** by Mistral AI was sufficient for all text and considered traits
- mistral-medium (version 2312) provided results comparable to OpenAI's GPT-4 at a similar cost to GPT-3.5 Turbo
- The Mixtral-8x7B and LLaMA 2 open source models did not provide satisfactory results
- However, the open source Mixtral-8x22B model released after the experiments provided comparable results to mistral-medium

#### Prompt design

**Binary Encoding of Categorical Traits:**
- Composed mostly of categorical traits
- Encoded in binary form, expressed as multiple choice textual questions
- Discover which trait values should be "1" or "0" for a given set of species and traits

**Encoding Process:**
1. Retrieve description sentences about a species
2. List all possible values for each trait (e.g., Life form, Phyllotaxis)
3. Prompt LLM to infer values based on provided text

**Example Prompt:**
- Ask about multiple traits and their values in one prompt
- Provide information about a single species only
- Some traits may not have evidence from the text

**Scaling Approach:**
- Repeat process for new species, adding more traits to the prompt
- Use a large enough context window or split task into multiple prompts

**Goal:**
- Obtain botanical trait information about Albizia coriaria using input text and dictionary of traits.

**Dictionary:**
```
{"Plant type": [ "Tree", "Shrub", "Bush", "Ficus strangler", "Liana", "Parasitic", "Palm tree", "Herbaceous" ], "Phyllotaxis": ["Phyllotaxis alternate", "Opposite phyllotaxis", "Phyllotaxis whorled"], "Trunk and root": ["Base of trunk straight", "Base of trunk flared", "Foothills", "Stilt roots", "Aerial roots"]}
```

**Instructions for LLM:**
1. Convert strings in values list to sublists (value, evidence)
2. Check association between trait name and value in text
3. Set evidence to 0 if not confident about association
4. Return dictionary of traits and their corresponding sublists

**Output Format:**
A JSON format containing a dictionary with all possible names and their respective binary evidence tuples.

### Evaluation metrics

**Evaluation Metrics for Automatic Trait Extraction**

**Comparison of Model Responses:**
- Compare automatic responses to manually curated expert matrices
- Report coverage rate: proportion of traits with values found
- Precision, recall, and F1 score calculated for detected traits
  * Precision: proportion correct predicted positives out of all predictions
  * Recall: proportion of manual dataset positives retrieved by approach
  * F1 score: geometric average of precision and recall

**Evaluation of False Negative Rate:**
- Assess whether false negatives arise from model or missing information
- Perform additional evaluation to quantify false negative rate
- Botanists assess relevance of text snippets to traits
  * Randomly selected trait and species from dataset
  * Text snippet with low distance in DistillBERT embedding space to trait name selected
  * Evaluate capacity of LLM extraction process by creating prompts for same pairs of sentences and traits as those presented to botanists
- Investigate if model excessively conservative or tends to hallucinate responses not explicitly in text.

## RESULTS

**Descriptive Text Classification Results**
* Approximately 1.45 million sentences generated through dataset creation process
* 1.1 million non-descriptive sentences, 356k descriptive ones
* **In-domain validation set**: high precision for both classes (Description: 0.96, Non-Description: 0.99) with F1 scores of 0.96 and 0.99 respectively.
* **Test set**: recall drops substantially for descriptive class from 0.95 to 0.55, while non-descriptive remains at 0.98 (precision: 0.94, recall: 0.98)
* Precision-recall metrics provided in Table 2

**Example Sentences and Their Corresponding Score:**
* Figure 4 illustrates a few example sentences with their corresponding description score.
* Model correctly identifies botanical descriptions as descriptive (Hedera helix an evergreen climbing plant, The leaves are alternate, etc.) while filtering out non-descriptive sentences (The house is large with enormous windows, This is something random) for further processing or storage in the database.

### Descriptive text harvesting

**Descriptive Text Harvesting:**
- Used trained and validated descriptive sentence detector
- Extracted species descriptions for downstream tasks
- Obtained text for:
  - Caribbean dataset: 40/42 species (35 sentences per species on average)
  - West Africa dataset: 358/361 species (36.8 sentences per species on average)
  - Palms dataset: 248/333 species (43.5 sentences per species on average)
- Couldn't extract text for some species in all datasets
- Internet domains contributing most descriptive sentences listed in Appendix.

### Automatic trait extraction

**Automatic Trait Extraction vs. Manually Curated Data Comparison**

**Coverage and F1 Scores**:
- Coverage ranges between 55% and 60%, meaning over half of traits are assigned values
- F1 scores range from 73% to 78%, with recall being remarkably constant at 77%-78%
- Precision varies between 70% in Palms to 80% in West Africa

**Per-Trait Performance**:
- Large variations in precision, recall, and F1 scores across traits within datasets
- Traits with fewer allowed values (e.g., life form) tend to have higher accuracy
- Traits with more possible values and ambiguities lead to more false positives

**Confusion Matrices**:
- Co-occurrence patterns between annotated data and predictions are similar, with reasonable mistakes
- For example, leaf position and fruit type co-occurrences are maintained, with opposite predictions being common
- Fruit color in Palms has many possible values, leading to more false positives

**Traits with Highest/Lowest F1 Scores**:
- Stem shape has high scores due to only two possible values and imbalanced distribution
- Leaf apex has low scores due to seven possible values and high overlap between them, leading to confusion

### Evaluation of the false negative rate

**Comparison of LLM and Botanists' Evaluations**

**Evaluation of False Negative Rate:**
- Estimating if coverage rate corresponds to actual data availability
- Comparing LLM responses with expert botanists on same sentences

**Confusion Matrix:**
- No strong bias towards over or under-detecting traits in text (Figure 7)
- Out of 1216 text samples, 24% contained relevant trait information by botanists, and the LLM reported found traits in 22%

**Agreement between LLM and Botanists:**
- High agreement for positive class: F1 score = 0.72
- High agreement for negative class: F1 score = 0.92
- Precision higher than recall suggests a conservative bias, with under-reporting being preferred over hallucination

**Comparison of LLM and Manually Curated Species-Trait Matrix:**
- Around 32% of traits labeled as "NA" by the LLM actually contained information in the text that was missed (Table 3)
- Performance of approach using whole input text and set of traits in a single prompt is roughly comparable to the manually curated species-trait matrix.

### Additional experimental results

**Experimental Results on Caribbean Dataset:**
* Two additional LLM settings evaluated: querying all traits simultaneously vs single trait
* No substantial degradation of results when querying all traits at once (Table 4)
* Co-occurrence patterns maintained between leaf position and fruit type (Figure 6)
* Improvement in coverage from 55% to almost 58% with larger input token requirements

**Comparison of Mistral Models:**
* Precision, recall, F1 scores, and coverage for different settings using mistral-medium model: single trait vs all traits (Table 4)
* Improved results with single trait querying in terms of precision, recall, and F1 score compared to all traits (0.7507 vs 0.7493 for Precision; 0.7920 vs 0.7800 for Recall; 0.7708 vs 0.7643 for F1)
* Mixtral-8x22B model provides comparable results with a 0.4% lower F1 score but improved coverage over 60% (Table 4)
* Running experiments with all traits on over 700 plant species required around $30 in Mistral AI credits.

## DISCUSSION

**Descriptive Text Harvesting**
- Majority of species yielded useful descriptive sentences
- Some species in Palms dataset returned no sentences at all
- Low recall compared to precision due to:
    - Study's focus on English-language HTML websites
    - Loss function accounting for label noise
    - Model determined that nearly half of text within descriptive sections did not genuinely pertain to descriptions
    - More concise and focused corpus result

**Automatic Trait Extraction**
- Proposed pipeline able to return trait values for over half of species-trait matrices (average F1 score > 0.75)
- Errors tend to be reasonable mistakes, with similar traits being confused for one another
- False negative rate evaluation shows LLM is well-balanced and has no strong tendency towards hallucinating or ignoring information
- Single trait per query results in very similar performance, indicating this behavior does not depend on the number of queried traits
- Low average coverage rate (about 55%) likely due to a lack of textual information rather than LLM unable to pick up the information
- Results using Mixtral-8x22B demonstrate approach could be reproduced and scaled to new species using openly available weights

**Limitations**
- Small number of plant species (approximately 700) evaluated, primarily woody plants from Europe and North America
- Focus on categorical traits; potential for adapting approach to other types of trait formulations in future work

**Concluding Remarks**
- Developed a pipeline using large language models to extract trait information from unstructured online text
- No manual annotations required for training, only initial list of species, traits, and possible values
- Results suggest potential for scaling this methodology to larger floras with more general lists of traits (e.g., those being developed by [Castellan et al., 2023])
- All code and data needed to reproduce results available at https://github.com/konpanousis/AutomaticTraitExtraction.

## Appendix
**Ground Truth Correlation Matrices and Co-occurrence Patterns**
**Fruit size**:
- **Palm dataset** correlation matrices:
  - Ground truth matrix (a): Fruit size within annotations
  - Predicted matrix (b): Fruit size between predictions and annotations
- Figure 8 shows co-occurrence matrices for some traits in the three datasets, comparing patterns of co-occurrence between annotations and predictions.

**Fruit color**:
- **Palm dataset** correlation matrices:
  - Ground truth matrix (c): Fruit color within annotations
  - Predicted matrix (d): Fruit color between predictions and annotations

**Stem shape**:
- **Western African dataset** correlation matrices:
  - Ground truth matrix (a): Stem shape within annotations
  - Predicted matrix (b): Stem shape between predictions and annotations
- Figure 9 shows co-occurrence matrices for stem shape and leaf apex, respectively the trait with the highest and lowest F1 scores.

**Source domains**:
- Figure 10 shows the top 20 Internet domains in terms of both URLs returned by the search API and total number of descriptive sentences they are the source of.

