# CypherBench Towards Precise Retrieval over Full-scale Modern Knowledge Graphs in the LLM Era

source: https://arxiv.org/html/2412.18702v1
by Yanlin Fengα Simone Papicchio, Sajjadur Rahman, Megagon Labs, Politecnico di Torino 

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Knowledge Graph Modeling in the LLM Era](#2-knowledge-graph-modeling-in-the-llm-era)
  - [2.1 Preliminaries: Knowledge Graphs, RDF and Property Graphs](#21-preliminaries-knowledge-graphs-rdf-and-property-graphs)
  - [2.2 Why is retrieval over modern KG hard?](#22-why-is-retrieval-over-modern-kg-hard)
  - [2.3 Hasn’t KBQA already solved KG retrieval?](#23-hasnt-kbqa-already-solved-kg-retrieval)
  - [2.4 Our Proposal: Property Graphs and Cypher as a Unified Interface](#24-our-proposal-property-graphs-and-cypher-as-a-unified-interface)
- [3 Transforming RDF to Property Graphs](#3-transforming-rdf-to-property-graphs)
  - [3.1 Domain-specific Schema Curation](#31-domain-specific-schema-curation)
  - [3.2 Automatic RDF-to-property-graph Transformation](#32-automatic-rdf-to-property-graph-transformation)
- [4 Constructing Questions](#4-constructing-questions)
  - [4.1 Graph Retrieval via Text-to-Cypher](#41-graph-retrieval-via-text-to-cypher)
  - [4.2 Preliminaries: Cypher Query Structure](#42-preliminaries-cypher-query-structure)
  - [4.3 Graph Pattern Design](#43-graph-pattern-design)
  - [4.4 Text-to-Cypher Task Generation](#44-text-to-cypher-task-generation)
  - [4.5 Question Rewriting and Verification](#45-question-rewriting-and-verification)
- [5 Evaluation Metrics](#5-evaluation-metrics)
  - [5.1 Execution Accuracy (EX)](#51-execution-accuracy-ex)
  - [5.2 Provenance Subgraph Jaccard Similarity (PSJS)](#52-provenance-subgraph-jaccard-similarity-psjs)
- [6 Experiments](#6-experiments)
  - [6.1 Evaluation Details](#61-evaluation-details)
  - [6.2 Main Results](#62-main-results)
  - [6.3 Performance Across Graph Matching Patterns](#63-performance-across-graph-matching-patterns)
  - [6.4 Performance Across RETURN Templates](#64-performance-across-return-templates)
  - [6.5 Performance Across Domains](#65-performance-across-domains)
  - [6.6 Error Analysis](#66-error-analysis)
- [7 Related Work](#7-related-work)
  - [7.1 KBQA and Graph Retrieval Methods](#71-kbqa-and-graph-retrieval-methods)
  - [7.2 Text-to-Query and KBQA Benchmarks](#72-text-to-query-and-kbqa-benchmarks)
  - [7.3 GraphRAG](#73-graphrag)
  - [7.4 Mapping RDF to Property Graphs](#74-mapping-rdf-to-property-graphs)
  - [7.5 Knowledge Graph Subsetting](#75-knowledge-graph-subsetting)
- [8 Conclusion](#8-conclusion)
- [Appendix A Additional Technical Details](#appendix-a-additional-technical-details)
  - [A.1 Graph Matching Patterns and RETURN Templates](#a1-graph-matching-patterns-and-return-templates)
  - [A.2 Question Rewriting Prompt](#a2-question-rewriting-prompt)
  - [A.3 Text-to-Cypher Prompt](#a3-text-to-cypher-prompt)
  - [A.4 Schema Fetching in Neo4j](#a4-schema-fetching-in-neo4j)
  - [A.5 Text-to-Cypher Error Taxonomy](#a5-text-to-cypher-error-taxonomy)
- [Appendix B Additional CypherBench Statistics](#appendix-b-additional-cypherbench-statistics)

## Abstract

**Retrieval from Graph Data**
- Crucial for augmenting large language models (LLM) with:
  - Open-domain knowledge
  - Private enterprise data
- Key component in the recent **GraphRAG system**

**Challenges with Modern RDF Knowledge Graphs**:
- Overly large schemas exceed typical LLM context window
- Use of resource identifiers
- Overlapping relation types
- Lack of normalization

**Proposed Solution**:
- **Property graph views** on top of underlying RDF graph
- Can be efficiently queried by LLMs using Cypher

**Implementation**:
- Instantiated on Wikidata
- Introduced **CypherBench**:
  - First benchmark with 11 large-scale, multi-domain property graphs
  - 7.8 million entities and over 10,000 questions

**Implementation Challenges**:
- Developing an RDF-to-property graph conversion engine
- Creating a systematic pipeline for text-to-Cypher task generation
- Designing new evaluation metrics

## 1 Introduction

**Graphs for Storing Entity-Relation Data:**
* Widely used in large-scale encyclopedic knowledge and domain-specific enterprise data storage
* Enable efficient processing of complex multi-hop aggregation queries
* Provide more compact representation of knowledge compared to raw textual documents
* Advantages have motivated research on knowledge graphs and KBQA

**Challenges in Retrieval from Modern RDF Knowledge Graphs:**
* Difficult for LLMs unlike success with relational databases
* Previous studies focused on simplified settings, limiting practical application
* Leading LLM frameworks prioritize retrieval from property graphs instead

**Root Cause and Proposed Solution:**
* RDF knowledge graphs have complex schemas and relationships
* Transforming them into multiple smaller property graphs simplifies retrieval
* Each graph functions as a domain-specific view, enabling efficient querying by LLMs
* Unified query language for both RDF graphs and property graph databases

**CypherBench:**
* Collection of 11 property graphs transformed from Wikidata
* Contains complete entities and relations from Wikidata that conform to a domain-specific schema
* Includes over 10,000 natural language questions for evaluation
* Global queries, multi-hop queries, temporal queries, and aggregation queries covered
* Significant challenges in benchmarking with current LLMs.

**Main Contributions:**
* Novel methodology to enable efficient text-to-Cypher retrieval over modern RDF knowledge graphs
* Collection of 11 large-scale property graphs for future research and knowledge source deployment
* RDF-to-property-graph transformation engine for Wikidata
* Text-to-Cypher / KBQA benchmark with over 10,000 instances covering various query types
* Automatic text-to-Cypher task generation pipeline
* Set of related tools including graph deployment Docker, evaluation scripts, and graph visualization tools.

## 2 Knowledge Graph Modeling in the LLM Era

### 2.1 Preliminaries: Knowledge Graphs, RDF and Property Graphs

**Knowledge Graphs: RDF vs Property Graph**

**Definition of Knowledge Graph:**
- List of relations (triples): subject entity, relation type, object entity
- Example: ("LeBron James", playsFor, "LA Lakers")
- Relation types are also called properties or predicates

**RDF:**
- Entities stored using Internationalized Resource Identifiers (IRIs)
- Entity properties stored as relations: subject = entity IRI, object = literal value
- Reification: create copy of relation as special entity linked to relation property using additional relation
- Popular knowledge graphs based on RDF: Wikidata, Freebase, DBpedia, queried using SPARQL

**Property Graph:**
- Entities and relations treated as objects
- Each object can be assigned types and have associated properties
- Access entities directly using names
- Property graph databases popular in industry: Neo4j
- No reification process used.

### 2.2 Why is retrieval over modern KG hard?

**Retrieval over Modern Encyclopedic Knowledge Graphs:**

**Challenges:**
- **Overly large schemas**: Modern knowledge graphs cover entities and relations across all domains, resulting in a massive schema that exceeds context window sizes of LLMs.
  - Wikidata: Over 4 million entity types, 12,000 relation types
  - RDF graphs allow arbitrary entities for subjects/objects
- **Use of resource identifiers**: SPARQL queries require identifiers obtained via external linkers.
  - Makes queries less readable
- **Semantically overlapping relation types**: Wikidata contains multiple relation types with similar meanings, causing confusion during retrieval.
  - Example: Six relation types for starting times of an entity
- **Lack of normalization**: RDF does not enforce type constraints or standardized units on values.
  - Resulting in incorrect types and varied units, leading to incorrect results during aggregation.

### 2.3 Hasn’t KBQA already solved KG retrieval?

**Challenges in Knowledge Base Question Answering (KBQA)**
- **Graph retrieval required**: answering questions using graphs instead of traditional databases or text corpora
- **Simplified settings used**: many studies focus on simplified settings, assuming entity and relation identifiers are provided
  - Reduces task to retrieval over a small local subgraph
- **Custom-designed intermediate logical forms**: lack support for certain graph querying functionalities
  - Examples: relation properties querying, grouping, variable-length path matching
- **Limited capabilities of existing KBQA approaches**
  - Struggle with queries involving:
    - Relation properties (time-sensitive queries)
    - Global queries without named entities
    - Complex aggregations over a large number of entities

### 2.4 Our Proposal: Property Graphs and Cypher as a Unified Interface

**Transforming RDF Graph into Property Graphs**
* Proposed solution: multiple domain-specific property graphs and Cypher for querying (Figure 2)
* Choice of Cypher: widespread adoption, efficient querying, and compatibility with LLM frameworks
* Property graphs serve as views on top of original RDF graph
* Each property graph contains data conforming to its schema
* Views can be updated when underlying RDF data changes
* Scalability without complex schemas or ambiguous relation types
* Transformation layer manages datatype conversion and unit standardization
* Efficient process: few seconds for small graphs (fewer than 10,000 entities)

**Wikidata Application**
* Demonstration of idea on Wikidata, the largest knowledge graph today
* Direct prompting baseline using gpt-4o achieves reasonable performance
* No external linkers or retrievers used.

**CypherBench Construction Process (Figure 2)**
* RDF data transformed into schema-enforced property graphs
* Efficient and accurate text-to-Cypher querying enabled
* Property graphs used to generate text-to-Cypher tasks.

## 3 Transforming RDF to Property Graphs

We transform RDF data into property graphs as the first step in building CypherBench. We chose Wikidata because it's the largest and most updated knowledge graph, containing 114 million entities and receiving 270 million edits from over 42,000 editors each month.

### 3.1 Domain-specific Schema Curation

**Domain and Property Graph Schema Curation**

**Selecting a Domain**:
- Identify entity and relation types and their properties relevant to the domain
- Explore Wikidata to find corresponding **QIDs (entity types)** and **PIDs (relation types and properties)**

**Schema Development**:
- Each property is assigned a **datatype**
- Properties representing quantities are given a **unit**, indicated in the property label (e.g., runtime_minute)
- Time investment: approximately 4 hours per graph on average

**Example Schema**:
- [Figure 3](https://arxiv.org/html/2412.18702v1#S3.F3 "Figure 3 ‣ 3.1 Domain-specific Schema Curation ‣ 3 Transforming RDF to Property Graphs ‣ CypherBench: Towards Precise Retrieval over Full-scale Modern Knowledge Graphs in the LLM Era")
- Complete schema available in [Appendix B](https://arxiv.org/html/2412.18702v1#A2 "Appendix B Additional CypherBench Statistics ‣ CypherBench: Towards Precise Retrieval over Full-scale Modern Knowledge Graphs in the LLM Era")

### 3.2 Automatic RDF-to-property-graph Transformation

**RDF-to-Property-Graph Engine Functions:**
* Issues SPARQL queries to Wikidata using curated schema identifiers
* Fetches all entities and relations conforming to the schema:
  + Filmmaker (Q11424) instances for award received (P166) relations
  + Award (Q618779) instances as objects of these relations
* Converts fetched data into receivesAward relations in target property graph

**Engine Functionalities:**
* **Datatype conversion**: enforces type constraints, converts values to str, int, float, date, list[str]
* **Date conversion**: retrieves precision using wikibase:timePrecision and keeps only fine-grained dates
* **Unit standardization**: enforces standardized units (e.g., centimeters) on property values representing quantities
* **Rank filtering**: fetches preferred relations for time-sensitive properties, normal or deprecated ones with additional properties
* **Selective entity fetching**: limits fetching to entities linked to certain relations to avoid out-of-memory issues

**Execution:**
* SPARQL queries executed against local Wikidata endpoint
* Results aggregated into final property graph
* Transformation time ranges from seconds to hours, depending on graph size

**Format and Deployment:**
* Property graph stored in DBMS-independent JSON format
* Custom Neo4j Docker image initialized with data from JSON file upon startup.

## 4 Constructing Questions

### 4.1 Graph Retrieval via Text-to-Cypher

**Text-to-Cypher Task Generation Pipeline**

**Step 1: Generating Initial (question, Cypher) pairs:**
- Input: graph schema and natural language question
- Output: executable Cypher query that returns the desired answer

**Patterns**:
- **Basic MATCH pattern**: queries for entities of a particular type
- **Special MATCH pattern**: queries for relationships between entities
- **RETURN template**: specifies the answer entity or entities to be returned

**Nodes**:
- Square nodes: all entities of a particular type
- Circular nodes: named entities
- Optional edges and nodes: denoted by dashed lines

**Step 2: Rewriting Questions**:
- Use a LLM to make questions sound more natural

**Patterns**:
- Provided in [Table 11](https://arxiv.org/html/2412.18702v1#A2.T11) and [Table 12](https://arxiv.org/html/2412.18702v1#A2.T12)

### 4.2 Preliminaries: Cypher Query Structure
A Cypher query starts with a MATCH clause, identifying matching subgraphs. Following this, other clauses (e.g. WHERE, WITH, ORDER BY, RETURN) perform transformations to generate the result.

### 4.3 Graph Pattern Design

**Graph Retrieval: Locating Relevant Subgraphs**
- **Core task**: locate subgraph relevant to query
- Balanced distribution of graph matching patterns
  - Template-based generation approach instead of crowd-sourcing
- Graph patterns categorized by isomorphism structure
  - Sample patterns in [Table 1](https://arxiv.org/html/2412.18702v1#S4.T1)
  - Complete notations in [Table 11](https://arxiv.org/html/2412.18702v1#A2.T11)
- Seven basic graph patterns: single answer node, up to two edges
- Five special graph patterns: comparison, grouping, optional matching, time-sensitive queries, union

**Comparison with Existing Benchmarks**
- Most KBQA benchmarks overlook global queries (GraphRAG targets)
- Global queries: no specific named entities
  - Listing queries like "List the names of all teams"
  - Complex queries like "Unique countries of citizenship for individuals in same movie"
- Answers depend on large number of documents and cannot be easily handled by standard RAG approaches.

### 4.4 Text-to-Cypher Task Generation

**Text-to-Cypher Question Generation**

**MATCH Clause Instantiation**:
- Create multiple Cypher **MATCH clause templates** by enumerating edge directions
- Pair each MATCH template with a human-written question template
- Instantiate the MATCH clauses by sampling:
  - Entity types for node variables (n, m0)
  - Relation types for edge variables (r0)
  - Entity names for named nodes (m0)
- Execute a special Cypher query to ensure non-empty results

**RETURN Clause Instantiation**:
- Pair each instantiated MATCH clause in **special categories** with dedicated RETURN clause templates
- Pair each RETURN clause in **basic categories** with one of 6 templates:
  - NAME, PROPERTY
  - SORT
  - WHERE
  - AGGREGATE, ARGMAX
- Instantiate the RETURN clauses by sampling properties, ranking orders, comparison operators, and aggregate functions

**Design Choices**:
- Ensure Cypher always returns literal values instead of node objects
- Allows benchmark to be used for evaluating non-Cypher graph retrieval methods

**Data Splits**:
- Questions split into training and test sets by domain
- 4 graphs allocated for training, 7 for testing
- Remove queries producing more than 10^5 records or taking more than 30 seconds to execute
- Demonstrate diverse and balanced distribution of patterns, templates, domains, and answer lengths

**Detecting Semantically Unrealistic Questions**:
- Blind sampling can result in semantically unrealistic questions
- Address this systematically by modeling:
  - **Cardinality**: one-to-one, one-to-many, many-to-one, many-to-many
  - **Participation**: total or partial (always/not always associated with the relationship)
  - **Entailment**: a relationship implying another

### 4.5 Question Rewriting and Verification

We use LLMs to rewrite template-generated questions into natural-sounding ones, while preserving entity names and values. This design choice allows us to evaluate LLMs directly without external linkers or retrievers. However, we observe that LLMs sometimes alter question meanings during rewriting. To address this, we implement three rounds of verification and revision using LLMs, followed by author inspection.

## 5 Evaluation Metrics

### 5.1 Execution Accuracy (EX)

**Text-to-SQL Evaluation Metric: Execution Accuracy**

**Definition:**
- Measures if results returned by predicted query match those of ground truth query
- Adapted from text-to-SQL literature for Cypher
- Borrows implementation from Spider leaderboard

**Execution Results Comparison:**
1. Identical tables: one can be transformed into the other through row and column permutations
2. Applicable to both Cypher and SQL queries
3. Objects in query results are serialized for comparison

**Calculation:**
- Compare execution results of ground truth (V) and predicted (q̂) Cypher queries
- Obtain final metric by averaging across all instances: EX(q,q̂)=1_V=V̂(V,V̂)

**Implementation:**
- Adapted from Spider leaderboard at [https://github.com/taoyds/test-suite-sql-eval]()

**Notes:**
- Cypher supports objects (lists and maps) in query results
- Execution accuracy can be applied to non-Cypher graph retrieval approaches as long as they return results in tabular format.

### 5.2 Provenance Subgraph Jaccard Similarity (PSJS)

**Graph Retrieval and Provenance Subgraph Jaccard Similarity (PSJS)**

**Core Task of Graph Retrieval**:
- Locate relevant subgraph using MATCH clause
- LLM may generate incorrect or partially correct MATCH clauses
- Resulting errors lead to zero execution accuracy

**Proposed Measure for Subgraph Matching Performance: Provenance Subgraph Jaccard Similarity (PSJS)**
1. **Definition**:
   - PSJS calculated as Jaccard similarity between provenance subgraphs of ground truth and predicted Cypher
2. **Provenance Subgraph**:
   - Obtained by pairing MATCH clause with RETURN *
   - Includes entities/nodes matched by the MATCH clause
3. **Example**:
   - For "Q18. What is the average longest lifespan of taxa that feed on Leporidae?":
      - Provenance subgraph includes Leporidae and all taxa feeding on it
4. **PSJS Calculation**:
   - G ∩ Ĝ / (G ∪ Ĝ)
   - G and Ĝ are provenance subgraphs of ground-truth and predicted Cypher, respectively
5. **Impact of Incorrect MATCH Clauses**:
   - Receives zero execution accuracy but non-zero PSJS score if some entities/nodes are correctly retrieved.

## 6 Experiments

### 6.1 Evaluation Details

**Evaluation of Zero-Shot Text-to-Cypher Performance**
* State-of-the-art LLMs tested on CypherBench: gpt-4o, gpt-4o-mini, gemini1.5, claude3.5-sonnet, and yi-large
* No use of training set for this study
* Models prompted with question, graph schema, and instruction (Appendix A.3)
* Cost per run: gpt-4o - 5.5, gpt-4o-mini - 0.3
* Execution on Neo4j using parallelization and timeout for metrics calculation

**Performance Metrics**:
- **Zero-shot execution accuracy (EX)**: not provided in table
- **Provenance subgraph jaccard similarity (PSJS)**
- **Executable percentage (Exec.)**

**Models Used**:
- gpt-4o and gpt-4o-mini: OpenAI's API
- gemini1.5 and claude3.5-sonnet: Google Cloud Vertex AI
- yi-large: Fireworks AI API

### 6.2 Main Results

**CypherBench Evaluation Results**
* **claude3.5-sonnet**: Execution accuracy of 61.58%, PSJS of 80.85%
* **gpt-4o**: Slightly worse than claude3.5-sonnet
* Best performing open-source model: 41.87% execution accuracy
* Models with fewer than 10B parameters: Less than 20% execution accuracy
* Difficulty of CypherBench: Highlights the challenges in graph matching, not just formatting errors
* Differentiating LLM capabilities: Smaller models within the same family perform significantly worse (e.g., gpt-, llama-, gemini-series)
* Effectiveness of CypherBench benchmarking: Figure 5 illustrates performance across basic and special MATCH patterns, RETURN templates, and domains.

### 6.3 Performance Across Graph Matching Patterns

**Performance Analysis of LLMs: gpt-4o, claude3.5-sonnet, qwen2.5-72b, llama3.1-70b**

**Focus Models**:
- gpt-4o
- claude3.5-sonnet
- qwen2.5-72b
- llama3.1-70b

**Evaluation Metrics**:
- Execution accuracy (EX)
- Path-based similarity judgment score (PSJS)

**Performance Trends**:
- All models exhibit near-perfect EX and low PSJS on basic categories
- Gradual decline in performance as graph patterns include more relations (PSJS)

**Error Analysis**:
- Most errors for basic category ![[Uncaptioned image]](https://arxiv.org/html/2412.18702v1/extracted/6093954/figures/match_patterns/basic_7.png) result from incorrect deduplication of distinct entities
- gpt-4o struggles with time-sensitive questions
- claude3.5-sonnet performs poorly on comparison questions

### 6.4 Performance Across RETURN Templates

The top right chart shows execution accuracy across different RETURN templates. Models have varying weaknesses depending on the template. Claude3.5-sonnet achieves near-zero accuracy on SORT questions because it includes unnecessary columns in its output, even though only entity names are requested.

### 6.5 Performance Across Domains

The bottom right chart in [Figure 5](https://arxiv.org/html/2412.18702v1#S6.F5 "Figure 5 ‣ 6.2 Main Results ‣ 6 Experiments ‣ CypherBench: Towards Precise Retrieval over Full-scale Modern Knowledge Graphs in the LLM Era") shows execution accuracy across domains. Four models exhibit similar trends, with flight_accident and nba being easiest, while performing similarly in other domains.

 Figure 6 shows error distributions made by gpt-4o and llama3.1-8b on 50 incorrect predictions.

### 6.6 Error Analysis

**Error Analysis of LLMs:**
* We examine gpt-4o and llama3.1-8b models to understand their error behavior
* Randomly sample 50 incorrect predictions from each model for annotation ([Appendix A.5](https://arxiv.org/html/2412.18702v1#A1.SS5))
* Define error categories during the annotation process (Table 4)

**Error Categories:**
* Reversed Direction: Model reverses direction of a relation
* Entity Linking: Entity name does not correspond to intended entity in database
* Pattern Not Aligned with Question: MATCH pattern conforms to schema but doesn't align with question's intent
* Schema Violation (llama3.1-8b): Superior schema following capabilities of gpt-4o are inferior in llama3.1-8b

**Example Predictions:**
[Table 4](https://arxiv.org/html/2412.18702v1#S6.T4) shows sample predictions from both models with annotated error categories.

**Error Distribution:**
[Figure 6](https://arxiv.org/html/2412.18702v1#S6.F6) illustrates the distribution of error categories across the models.

## 7 Related Work

### 7.1 KBQA and Graph Retrieval Methods

**CypherBench: Graph Retrieval Methods Related to Knowledge Base Question Answering (KBQA)**

**Background:**
- CypherBench serves as benchmark for evaluating KBQA and graph retrieval methods

**Types of Methods:**
1. **Approximate Retrieval Methods**: Identify top relevant elements based on relevance to the question
   - Retrieve k-hop neighborhood of mentioned entities
   - Verbalize entities or relations into text, use embedding-based methods
2. **Limitations:**
   - Inability to handle large number of entities (global queries or complex aggregation queries)
   - Reliance on expensive in-memory operations or need to embed entire graph

**Precise Retrieval Methods:**
1. **Description:**
   - Translate question into formal language query that fetches exactly what it asks for
2. **Custom Logical Forms:**
   - Transpiled into actual database queries or executed by custom engine
3. **Limitations:**
   - Lack support for certain graph querying features like grouping and variable-length path matching
   - Inability to handle large graphs due to scalability issues and limitations compared to standard database query languages
4. **Recent Works:**
   - Use LLMs to generate graph database queries (e.g., SPARQL or Cypher)
   - Assumption of identified entities in the question
5. **Limitations:**
   - Simplifications such as assuming identifiers are provided
   - Working with smaller graphs only

**Our Work:**
- Handles realistic scenario of graph retrieval over full-scale knowledge graphs using only the question as input.

### 7.2 Text-to-Query and KBQA Benchmarks

**Text-to-Query Benchmarks Comparison**

**CypherBench vs. Existing Text-to-Query Benchmarks**:
- **CypherBench**: text-to-query benchmark with databases and (question, database query) pairs
- **KBQA benchmarks**: represent specific type of text-to-query benchmarks where the databases are knowledge graphs, and queries are graph database queries
- Comparison in [Table 6](https://arxiv.org/html/2412.18702v1#S7.T6):
    - **Schema Size** and **Data Size**: insights into the complexity of databases in existing text-to-query benchmarks
    - Most existing KBQA benchmarks are based on text-to-SPARQL over RDF knowledge graphs, but the massive schema poses challenges for LLMs in zero-shot settings
    - CypherBench has a comparable schema size to text-to-SQL benchmarks, while still encompassing up to 7 million entities

**Text-to-SQL Benchmarks**:
- **Spider**: based on Wikipedia and other sources, with 200 graphs, schema size of 5.1 tables and 27.6 columns, and data size of 400k rows
- **BIRD-SQL**: based on Kaggle and other sources, with 95 graphs, schema size of 7.3 tables and 54.2 columns, and data size of 52M rows

**Text-to-SPARQL/RDF Benchmarks**:
- **LC-Quad 2.0**: based on Wikidata, with 1 graph, schema size of 12k relation types, and data size of 114M entities
- **GrailQA**: based on Freebase, with 1 graph, schema size of 37k relation types, and data size of 45M entities
- **KQA Pro**: based on FB15k-237, with 1 graph, schema size of 0.8k relation types, and data size of 16k entities

**Text-to-nGQL/Property Graphs Benchmarks**:
- **R^3 -NL2GQL**: based on OpenKG, with 3 graphs, schema size of 5.3 relation types and 13 properties, and data size of 46k entities
- **Fin/Medi-GQL**: based on OpenKG, with 2 graphs, schema size of 13 relation types and 38 properties, and data size of 713k entities

**Text-to-Cypher/Property Graphs Benchmarks**:
- **MetaQA-Cypher**: based on OMDb, with 1 graph, schema size of 5 relation types and 5 properties, and data size of 43k entities
- **SpCQL**: based on OwnThink, with 1 graph, schema size of 480k relation types and 1 property, and data size of 16M entities

### 7.3 GraphRAG

**Microsoft's GraphRAG**:
- Introduced to address corpus-level summarization queries (e.g., "What are the main themes in the dataset?")
- Cannot be handled by standard top-k embedding-based retrieval methods
- Leverages a **centralized knowledge graph** to index textual documents
- Enables handling of queries that rely on a large volume of documents

**GraphRAG Stages**:
- **Knowledge graph construction** during indexing time
- **Graph retrieval** during query time

**Comparison with Original GraphRAG**:
- Original GraphRAG uses a slightly different graph formalism than typical knowledge graphs
- Nodes are **entity communities at various abstraction levels**
- Retrieval performed by fetching all communities at a specific level

**LlamaIndex and Property Graph Index**:
- Leading open-source LLM framework for RAG workflows
- Constructs a **Neo4j property graph** from textual documents using LLMs during indexing time
- Conducts graph retrieval via **text-to-Cypher** during query time

**Text-to-Cypher Benchmark**:
- Provides the first comprehensive benchmark for evaluating graph retrieval, a critical component in GraphRAG.

### 7.4 Mapping RDF to Property Graphs

**RDF to Property Graph Transformations**

**Methods from Semantic Web Community:**
- Several studies have explored methods for transforming RDF graphs into property graphs [^57]<sup class="ltx_note_mark">16</sup><sup class="ltx_note_mark">16</sup>16
- Transformations can be computationally expensive for full-size modern RDF graphs like Wikidata.

**Two-step Process:**
- First, RDF triples are mapped directly to edges in the property graph [^57]
- Second, edges representing entity properties are transformed into node properties

**Exception:**
- [g2glab/g2g](https://github.com/g2glab/g2g) adopts an approach similar to this by transforming RDF into property graphs through executing SPARQL queries over RDF.

**Lacking Key Functionalities:**
- Their method lacks essential functionalities for ensuring output quality described in [section 3](https://arxiv.org/html/2412.18702v1#S3)

### 7.5 Knowledge Graph Subsetting

Several tools have been developed to extract domain-specific subgraphs of Wikidata or general RDF knowledge graphs to address scalability issues in modern knowledge graphs. Examples include KGTK, WDumper, and WDSub. However, these tools process the entire RDF dump and produce RDF output.

## 8 Conclusion

Wikidata has received over 2 billion edits from 42,000 active editors worldwide, making it a comprehensive knowledge source. Our study proposes techniques for integrating Wikidata with Large Language Models (LLMs), offering opportunities in knowledge graphs and graph retrieval research.

## Appendix A Additional Technical Details

### A.1 Graph Matching Patterns and RETURN Templates

The graph matching patterns and RETURN templates are listed in [Table 11](https://arxiv.org/html/2412.18702v1#A2.T11) and [Table 12](https://arxiv.org/html/2412.18702v1#A2.T12).

### A.2 Question Rewriting Prompt

The prompt to rewrite template-generated questions is in [Table 9](https://arxiv.org/html/2412.18702v1#A2.T9).

### A.3 Text-to-Cypher Prompt

The prompt for evaluating LLMs is shown in [Table 10](https://arxiv.org/html/2412.18702v1#A2.T10) of CypherBench statistics.

### A.4 Schema Fetching in Neo4j

**Retrieving Graph Schema from Neo4j Database:**
* Use custom queries instead of db.schema.visualization and apoc.meta.data for large graphs
* Queries provide complete and accurate schemas deterministically
* Approximately 15 times slower than built-in procedures (apoc.meta.data)
* Retrieve schemas using the following queries:

1. **Retrieve Nodes**:
   - Labels, properties, incoming and outgoing relationships
2. **Retrieve Relationships**:
   - Types, properties, source and target nodes
3. **Aggregate and Serialize Results**
4. **Example Query:**
   - Custom Cypher query for node retrieval (modify according to your requirements)

### A.5 Text-to-Cypher Error Taxonomy

There is no need to repeat the passage as it's not there. The original message had a link and instructions to refer to Table 13, but since I'm unable to access external links or tables, I can provide you with a summary:

Table 13 provides detailed definitions of each error category in CypherBench, including examples.

## Appendix B Additional CypherBench Statistics

**CypherBench Statistics**
- **Table 8**: Statistics of the graphs
  - Wikipedia refers to the number of English Wikipedia articles linked from the entities (roughly the number of entities with Wikipedia articles)

**Question Rewriting Prompt and Text-to-Cypher Prompt**
- Table 9: Sample question rewriting prompt
- Table 10: Sample text-to-Cypher prompt used in experiments

**Graph Matching Patterns**
- Table 11: Sample questions with various graph matching patterns from the benchmark
  - Nodes in purple denote the **answer entities**
  - Square nodes ([[Square Grey]](https://arxiv.org/html/2412.18702v1/extracted/6093954/figures/match_patterns/square_gray.png), [[Square Color]](https://arxiv.org/html/2412.18702v1/extracted/6093954/figures/match_patterns/square_color.png)) denote all entities of a particular type
  - Circular nodes ([[Circle Grey]](https://arxiv.org/html/2412.18702v1/extracted/6093954/figures/match_patterns/circle_gray.png) [[Circle Color]](https://arxiv.org/html/2412.18702v1/extracted/6093954/figures/match_patterns/circle_color.png)) represent named entities
  - Nodes and edges with dashed lines ([[dashed lines]](https://arxiv.org/html/2412.18702v1/extracted/6093954/figures/match_patterns/dashed.png)) are optional
  - Edges with diamond arrowheads ([[diamond]](https://arxiv.org/html/2412.18702v1/extracted/6093954/figures/match_patterns/diamond.png)) indicate relations with time sensitivity constraints

**RETURN Clause Categories and Sample Questions from the Benchmark**
- Table 12: RETURN clause categories and sample questions from the benchmark
  - RETURN clauses shown here apply to any graph pattern in the basic categories

**Text-to-Cypher Error Categories**
- Table 13: Definitions and examples for the 10 text-to-Cypher error categories

**Schemas of the 11 Graphs in the Benchmark**
- Table 14: Schemas of the 11 graphs in the benchmark
  - Color of property boxes indicates whether they are entity properties (e.g., ) or relation properties (e.g., )

**Schemas of the 11 Graphs in the Benchmark (Continued)**
- Table 15: Schemas of the 11 graphs in the benchmark (Continued)

