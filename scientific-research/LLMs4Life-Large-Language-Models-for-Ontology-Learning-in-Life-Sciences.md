# LLMs4Life Large Language Models for Ontology Learning in Life Sciences

source: https://arxiv.org/html/2412.02035v1
by Nadeen Fathallah, Alsayed Algergawy

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 Methodology](#3-methodology)
- [4 Experiments and Results](#4-experiments-and-results)
  - [4.1 Experiment 1: Baseline NeOn-GPT (AquaDiva)](#41-experiment-1-baseline-neon-gpt-aquadiva)
  - [4.2 Experiment 2: Count Metric-Guided Prompts (AquaDiva)](#42-experiment-2-count-metric-guided-prompts-aquadiva)
  - [4.3 Experiment 3: Merging Ontologies (AquaDiva)](#43-experiment-3-merging-ontologies-aquadiva)
  - [4.4 Experiment 4: Re-prompting \& Advanced Role-play Prompting (Habitat)](#44-experiment-4-re-prompting--advanced-role-play-prompting-habitat)
  - [4.5 Experiment 5: Reuse (Role)](#45-experiment-5-reuse-role)
  - [4.6 Experiment 6: Reuse of domain-specific examples (Carbon \& Nitrogen Cycling)](#46-experiment-6-reuse-of-domain-specific-examples-carbon--nitrogen-cycling)
  - [4.7 Comprehensive Ontology Performance Overview](#47-comprehensive-ontology-performance-overview)
- [5 Conclusion and Future work](#5-conclusion-and-future-work)
- [6 Appendix A: Persona Used for Role-play Prompting](#6-appendix-a-persona-used-for-role-play-prompting)

## Abstract

**Challenges in Ontology Learning for Complex Domains using LLMs:**
* Existing Large Language Models (LLMs) have limitations:
	+ Struggle to generate ontologies with multiple hierarchical levels
	+ Limited rich interconnections
	+ Inadequate domain adaptation
* Reasons for these challenges:
	+ Token constraints in LLMs
	+ Insufficient specialized knowledge

**Addressing Challenges through NeOn-GPT Pipeline Extension:**
* Enhancing generated ontologies' domain-specific reasoning and structural depth with advanced prompt engineering techniques
* Ontology reuse for improved results

**Evaluation of LLMs in Complex Domains:**
* Case study: AquaDiva ontology in the life science domain (AquaDiva <sup class="ltx_note_mark">1</sup><sup class="ltx_note_mark">1</sup>[https://www.aquadiva.uni-jena.de/](https://www.aquadiva.uni-jena.de/))
* Evaluation criteria: logical consistency, completeness, scalability

**Results and Conclusion:**
* LLMs can be viable for ontology learning in specialized domains like life science
* Addresses limitations in model performance and scalability.

## 1 Introduction

**Ontology Learning**

**Tasks**: Ontology extraction, ontology generation, or ontology acquisition.
- Automatic/semi-automatic creation of ontologies from natural language text
- Extracting domain terms and relationships between concepts
- Encoding with an ontology language for easy retrieval

**Challenges in Complex Domains**:
- Limited ability of Large Language Models (LLMs) to generate ontologies in highly specialized domains like life sciences
- Inherent complexity, domain-specific terminologies, and data limit logical depth required for advanced reasoning

**AquaDiva Ontology as a Use Case**:
- Collaborative research project across biology, geology, chemistry, and computer science
- Objective: Enhance understanding of Earth's critical zone
- Standardize data, integrate, and ensure interoperability using semantic web approaches
- AquaDiva Ontology (ADOn): Developed with 78.840 axioms, 8.892 concepts, and 245 object properties

**Importance of Accurate Ontologies**:
- Facilitate scientific research
- Support advanced data analysis and decision-making
- Enhance understanding of complex ecological processes
- Improve scientific communication

**Limitations of Current Approaches**:
- Heavy reliance on manual processes: labor-intensive, prone to human error
- Potential for efficiency enhancement with LLMs but requires rigorous evaluation
- Evaluating generated ontologies for logical soundness, domain coverage, and adaptability

**Evaluation of LLMs in Complex Domains**:
- Insufficient domain adaptation leads to simplified structures, shallow hierarchies, and limited subclass depth
- Vast amount of information often exceeds token limitations, resulting in incomplete outputs

**Proposed Approach**:
- Extension of NeOn-GPT pipeline with advanced prompt engineering techniques
  - Re-prompting strategies to refine output and enhance depth and hierarchy
  - Increased use of few-shot examples and advanced role-play prompting
  - Domain categorization strategy to handle token limitations
- Incorporation of ontology reuse in the NeOn-GPT pipeline

**Experimental Evaluation**:
- Assess structural complexity, depth, and logical consistency of generated ontologies using AquaDiva ontology as a case study.

## 2 Related Work

**LLMs in Ontology Learning:**
* LLMs enhance various ontology-related tasks: creation, enrichment, refinement (Mateiu et al.)
* Challenges: maintaining deep structure, avoiding irrelevant axioms (Mateiu et al.), fine-tuning necessary for domain-specific tasks (Babaei Giglou et al.)
* LLMs4OL framework: Term Typing, Taxonomy Discovery, Non-Taxonomic Relation Extraction (Babaei Giglou et al.)
* Reduces human effort but requires manual validation due to variability in LLM output and prompt sensitivity (Kommineni et al., Saeedizade and Blomqvist)
* Shallow hierarchy generation, token limitations, insufficient domain adaptation are common limitations of off-the-shelf LLMs (Mai et al.)

**Improvements:**
* Employ re-prompting techniques and keyword categorization strategy to manage token constraints (this work)
* Leverage ontology reuse: incorporate existing ontological structures to guide LLM generation of more detailed hierarchies and relationships (this work)
* Ensures consistency with established domain knowledge while allowing for comprehensive ontology generation in specialized domains like AquaDiva ontology.

## 3 Methodology

**NeOn-GPT Pipeline for Ontology Learning: Extensions for Complex Domains**

**Background:**
- Built on previous work with NeOn methodology framework [^11]
- Translates structured process into prompts for pre-trained LLMs
- Effective ontology generation in popular domains but not specialized ones

**Motivation:**
- Extend pipeline to handle more complex and specialized domains, such as life sciences
- Enhancements enable deeper understanding of domain-specific knowledge

**Methodology:**
1. **Specification of Ontology Requirements:**
   - Define purpose, scope, target group, functional requirements using chain-of-thought (CoT) prompting
   - Integrate domain descriptions and keywords into CoT prompt
   - Refine role-play persona for more contextually relevant outputs
2. **Reuse of Ontological Knowledge Resources:**
   - Identify critical limitations in LLM's ability to generate ontological structures that meet predefined criteria
   - Introduce reuse of structural information (count metrics) from gold standard AquaDiva ontology to improve overall structure and alignment with predefined metrics
3. **Ontology Conceptualization:**
   - Extract entities and relationships through few-shot prompting
   - Tailor process to AquaDiva ontology using domain-specific examples

**Enhancements:**
1. **Reuse (Role):**
   - Manually extract examples from Environment Ontology (ENVO) for assessment of hierarchical depth, interoperability, and relevance within broader ontological ecosystem
2. **Structural Improvements:**
   - Prompt LLM to target predefined counts for various ontology components
   - Introduce refined prompt to increase subclass count and improve hierarchy depth and interconnectivity

**Benefits:**
- Effectively generates ontologies in complex domains, significantly advancing ontology learning for niche areas.

## 4 Experiments and Results

**Evaluating LLM's Performance with AquaDiva Ontologies:**
* Experiments to assess impact of pipeline updates on generating complex life science ontologies
* Focusing on AquaDiva, a domain integrating hydrogeology, microbial ecology, etc. (78.840 axioms, 8.892 concepts, 245 object properties) [^12]
* Experiments conducted using GPT-4o [^17]
* Results discussed to illustrate improvements achieved through updated pipeline
* Code base accessible at: [https://github.com/NadeenAhmad/NeOn-GPTAquaDivaOntology](https://github.com/NadeenAhmad/NeOn-GPTAquaDivaOntology)

**Preparing for Experiments:**
* Evaluating LLM performance using AquaDiva ontologies before and after pipeline updates
* Focus on complex life science domains, specifically AquaDiva
* Experiments conducted with GPT-4o [^17]
* Results discussed to demonstrate improvements from updated pipeline
* Code base available at: [https://github.com/NadeenAhmad/NeOn-GPTAquaDivaOntology](https://github.com/NadeenAhmad/NeOn-GPTAquaDivaOntology)

**Assessing LLM Performance:**
* Evaluating Logical Consistency and Structural Depth of Generated Ontologies
* Using AquaDiva ontologies as a test domain (complex life science domain)
* Experiments conducted with GPT-4o [^17]
* Results presented to illustrate improvements from updated pipeline.

### 4.1 Experiment 1: Baseline NeOn-GPT (AquaDiva)

**Experiment 1: Baseline NeOn-GPT (AquaDiva)**
* **Evaluated LLM performance without enhancements**: Applied original pipeline, included domain-specific keywords to compensate for lack of relevant training data
* **Results:**
  + Captured key concepts like 'aquifers' and 'microbial communities' but:
    - Ontology remained overly simplistic with sparse hierarchy
    - Lacked complexity needed for advanced ecological modeling
  + Metrics and class hierarchy:
    - 176 classes (significantly fewer than gold standard)
    - 44 object properties (omission of crucial relationships and subclass hierarchies)
    - Absence of equivalent and disjoint classes, reduced logical axioms (323 vs. 16,303 in the gold standard)
* **Generated ontology:**
  + Included important concepts like 'Aquifer' and its subclasses, environmental factors
  + Lacked relational depth to describe interactions between entities, impacting ability to model microbial interactions within environments
  + Limited representation of complex ecological relationships and taxonomic structures due to:
    - Reduced number of individuals (13)
    - Data properties (26)
* **Impact on utility**: Simplified logical framework made it difficult to support detailed ecological queries, significantly reducing its utility for in-depth reasoning about environmental and biological phenomena.

### 4.2 Experiment 2: Count Metric-Guided Prompts (AquaDiva)

**Experiment 2: AquaDiva Ontology Generation (Count Metric-Guided Prompts)**
* Revised prompt pipeline from Experiment 1 to incorporate explicit count metrics
* Incorporated AquaDiva gold standard metrics: classes (8,892), object properties (245)
* Emphasized subclass count of n-1 (where n is the total number of classes) to address shallow hierarchies in Experiment 1

**Results:**
* More interconnected structure with increased density and layered hierarchy
* Increased concepts and relationships, aligning more closely to domain complexity
* Significant improvements over initial version
* Notable increase in classes (342) and axioms (795) compared to Experiment 1
* Improved hierarchy with more subclass levels, e.g., "HydroChemistry" -> "SubClassOf" -> "Geological Chemistry" -> "SubClassOf" -> "Earth Science"
* Discrepancies in object property count (8 vs. expected 245) due to:
	+ GPT-4's output limit (4096 tokens) restricting content generation
	+ LLM's mathematical limitations, particularly in precise counting tasks
	+ Redundancy with overlapping object properties, e.g., "interact with" and "interacts with," requiring further refinement.

### 4.3 Experiment 3: Merging Ontologies (AquaDiva)

**Experiment 3: Merging Ontologies (AquaDiva)**
* Improvements in key metrics:
  + Total axiom count increased to 1,479
  + Object property count rose to 50
* Captured broader set of relationships and axioms, resulting in more comprehensive ontology
* Limitations persist:
  + Class count (500) below gold standard AquaDiva ontology
  + Discrepancies in data and annotation properties
  + Object property count (245) still falls short of expected value
* Progress made in logical consistency:
  + 713 logical axioms
  + 114 SubClassOf axioms
* Improved structure for defining relationships and hierarchical taxonomies
* Number of disjoint classes (109) lagging, impacting ability to differentiate categories for accurate environmental modeling.

### 4.4 Experiment 4: Re-prompting & Advanced Role-play Prompting (Habitat)

**Experiment 4: Generating an Ontology for Habitat Category (AquaDiva)**

**Approach**: Instructed LLM to categorize AquaDiva keywords into 22 categories instead of generating the entire ontology due to output token constraints.

**Goals**:
- Improve quality and precision by providing richer domain-specific context
- Increase number of few-shot examples for better guidance
- Refine role-play persona as an expert aquatic ecologist
- Apply re-prompting for iterative refinement

**Results**:
- Ontology metrics: 630 axioms, 275 logical axioms, 75 classes
- Progress in object property count (47) but still lacking in several areas
- Incomplete class relationships (single DisjointClasses axiom, 3 EquivalentClasses axioms)
- Insufficient SubClassOf axioms (44) for detailed hierarchical structure
- Lack of comprehensive disjointness and equivalence axioms.

### 4.5 Experiment 5: Reuse (Role)

**Experiment 5: Role Ontology Generation (Role)**

**Strengths:**
- **Axiom count**: 969 axioms
- **Class count**: 118 classes
- **Subclass count**: 86 subclasses (significant increase from Experiment 4)
- Represents complex relationships within the Role domain
- Suitable for supporting advanced reasoning tasks
- Includes 57 individual instances

**Improvements:**
- Enhanced subclass hierarchy through reuse of ENVO example
- More layered and detailed ontology structure

**Limitations:**
- **Logical consistency**: Needs improvement with only 17 EquivalentClasses axioms
- Underdeveloped in terms of DisjointClass distinctions, with only 10 axioms
- Broadness of some classes (e.g., "Biological Role", "Chemical Role") may dilute focus and reduce utility within AquaDiva ontology.

### 4.6 Experiment 6: Reuse of domain-specific examples (Carbon & Nitrogen Cycling)

**Experiment 6: Carbon and Nitrogen Cycling Domain**

**Ontology Generation**:
- Building on lessons from previous experiments
- Reuse of existing ontological resources improves terminology generation
- Increased number of classes and subclasses from [4.4](https://arxiv.org/html/2412.02035v1#S4.SS4) to [4.5](https://arxiv.org/html/2412.02035v1#S4.SS5)
- Selected Carbon and Nitrogen Cycling domain for evaluation

**Improvements**:
- Continued using advanced role-play persona from Experiments 4 & 5
- Detailed description with domain-specific keywords to guide model's understanding
- Increased number of few-shot examples tailored to Carbon and Nitrogen Cycling domain
- Syntax and consistency restrictions at all stages for logical consistency

**Reuse of Existing Ontological Resources**:
- Targeted reuse approach, using specific components from ENVO
- Clearer structure, ensuring accurate hierarchical depth and detailed relationships

**Results**:
- Significant improvements in capturing complex biochemical processes
- Key entities like "Carbon Fixation" and "Nitrogen Transformation" accurately modeled
- 157 classes, 63 object properties, enabling detailed interactions
- Hierarchical depth enhanced with 130 SubClassOf axioms from ENVO
- 1,169 axioms, 455 of which are logical, for more detailed process representations
- Limited ability to fully capture equivalent biochemical processes and distinctions between exclusive pathways.

### 4.7 Comprehensive Ontology Performance Overview

**Comparative Analysis of Generated Ontologies**

**Evaluation Metrics:**
- Number of entities in LLM-generated ontologies that match entities in gold standard ontologies
- Concept similarity: average similarity score for matched concepts with gold standard ontology

**Results with AquaDiva Ontology:**
| Experiment | Matched Entities | Average Similarity Score |
|---|---|---|
| 1 (Baseline) | 17 | 0.896 |
| 2 (Count Metric-Guided Prompts) | 66 | 0.894 |
| 3 (Merging Ontologies) | 80 | 0.874 |
| 4 (Re-prompting & Roleplay Prompting) | 16 | 0.898 |
| 5 (Reuse) | 56 | 0.905 |
| 6 (Reuse of domain-specific examples) | 65 | 0.859 |

**Results with ENVO Ontology:**
| Experiment | Matched Entities | Average Similarity Score |
|---|---|---|
| 1 (Baseline) | 8 | 0.877 |
| 2 (Count Metric-Guided Prompts) | 57 | 0.969 |
| 3 (Merging Ontologies) | 60 | 0.885 |
| 4 (Re-prompting & Roleplay Prompting) | 13 | 0.800 |
| 5 (Reuse) | 54 | 0.886 |
| 6 (Reuse of domain-specific examples) | 51 | 0.884 |

**Findings:**
- Generated ontologies do not fully capture breadth and depth of domain knowledge as gold standard ontologies
- Aligned entities demonstrate high similarity scores, approaching or exceeding 0.85
- Number of matched entities increases across experiments, indicating improvements in prompt engineering techniques and pipeline refinements.
- LLM-based approaches show potential for complex ontology generation tasks.

## 5 Conclusion and Future work

**Approach to Enhance Ontology Learning in Complex Domains:**
- Extends NeOn-GPT pipeline for deep and well-structured ontologies in complex domains like life sciences
- Addresses limitations of Language Models (LLMs) in generating complex hierarchies and token constraints
- Leverages advanced prompt engineering, ontology reuse, and iterative refinement

**Challenges:**
- Shallow hierarchies: addressed with careful prompt design and curated examples for reuse
- Token constraints: not specified in the provided text how they are tackled

**Case Study: AquaDiva**
- Complex domain requiring additional contextual information in prompts and carefully curated examples
- Quality improvement through manual efforts and expert input

**Future Work:**
- Explore automating the process using Retrieval-Augmented Generation (RAG)
- Integrate external domain-specific resources dynamically to reduce manual intervention
- Evaluate complete AquaDiva ontology, focusing on refining consistency in relationships and capturing intricacies of specialized domains.

**Acknowledgements:**
- Funding by Deutsche Forschungsgemeinschaft (DFG) as part of CRC 1076 AquaDiva (Projectnumber 218627073).

## 6 Appendix A: Persona Used for Role-play Prompting

**Expert Aquatic Ecologist and Knowledge Engineer**

**Background**:
- PhD in Ecology
- Additional training in data science and semantic technologies
- Extensive experience in field research and computational modeling of aquatic ecosystems

**Specialties**:
- Understanding biological, chemical, and physical characteristics of water bodies
- Developing ecological ontologies for scientific research and environmental management
- Identifying essential entities and relationships within the ecological domain (e.g., key species, roles, conditions, processes)
- Applying tools like Turtle to create well-defined ontologies representing complex ecological data in a structured format
- Meticulous and user-centric approach to ontology creation, ensuring interoperability, data sharing, and reuse among stakeholders

**Expertise**:
- Deep domain knowledge of aquatic ecology
- Enhancing understanding and application of ecological data through detailed explanations of concepts and interconnections
- Bridging the gap between raw data and actionable knowledge by developing comprehensive ontological frameworks for advanced analysis and decision-making in aquatic ecology

**AquaDiva Domain**:
- Studying groundwater ecosystems
- Integrating hydrogeology, microbial ecology, geochemistry, karst systems, and environmental science.
