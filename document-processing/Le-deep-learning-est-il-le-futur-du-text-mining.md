# Is Deep Learning the future of text mining

published: 2016-09-02
source: https://scoms.hypotheses.org/657
by Pierre-Carl Langlais

## Contents
- [The art of translating words into vectors...](#the-art-of-translating-words-into-vectors)
- [Towards an exploration of the semantic space of the text?](#towards-an-exploration-of-the-semantic-space-of-the-text)
- [Test of release of semantic sets](#test-of-release-of-semantic-sets)
- [Some perspectives](#some-perspectives)

## Le deep learning est-il le futur du text mining

**Key Concepts and Technologies**
- **Deep learning**: Current technological expectation
- **Big data**: Often conceived alongside deep learning
- **Artificial neuron networks**: Revolutionizing various fields
- **Deep Dream**: App creating strange dream compositions

**Applications of Deep Learning**
- Image recognition
- Mastering complex games (e.g., Go)
- Automated writing
- Sound and image processing

**Text Mining and Digital Humanities**
- **Text mining**: Growing field
- **Common techniques**: Correspondence analysis, latent semantic analysis, SVMs
- **Innovation in digital humanities**: Applying algorithms to large text sets
- **Study by Underwood and Sellers**: Focuses on "literary standards" in 19th-century English journals

**Word2Vec**
- Introduced in 2013 by **Tomas Mikolov** et al.
- Not strictly deep learning, but opens new prospects
- Challenging to implement practically
- Used in digital humanities projects (e.g., tracing specialized terms in Vogue magazine)

**Experimental Corpus**
- **Hector Berlioz's columns**: Journal of Debates (1834-1861)
- Nearly 400 articles, over 1 million words

**Tools and Methods**
- **Gensim**: Python extension for Word2Vec
- Focus on exploration and experimentation
- Code not published due to ongoing development

**Key Points**
- Deep learning not yet fully integrated into remote reading toolbox
- Word2Vec shows promise for textual and literary studies
- Emphasis on raising questions rather than providing solutions
- Data file generated by Word2Vec available on Github for further exploration
### The art of translating words into vectors...

**Word2Vec Transformation:**
- Word2Vec transforms words into vectors (number lists), e.g., "Beethoven": -0.073912 -0.077942 ...
- This transformation reduces the contextual relationship between words, similar to file size reduction techniques like Huffman coding.
- The resulting vectors carry information about the relationship between terms but cannot be reverse engineered to reconstruct the text.

**Word Embedding:**
- Word embedding is a concept introduced in 2003 that focuses on the connection between words and texts.
- It represents words as vectors in a high dimensional space, preserving their contextual relationships within a corpus.
- This process allows for new insights into relationships between terms beyond traditional text mining methods.

**Applications:**
- Identifying words closest to a given term: "Beethoven" -> ["mozart", 0.7579751014709473]
- Finding related terms based on their usage in context, e.g., "Fidelio" is to "Beethoven" as "Freisch-tz" is to "Weber".

**Limitations:**
- Variability in results due to model configurations and context size.
- Difficulty maintaining relationships between terms when removing specific contexts, like operas or countries (Mikolov et al.).

### Towards an exploration of the semantic space of the text?

**Word2Vec: Text Analysis using Artificial Neuron Networks**

**Overview:**
- Word2Vec employs artificial neuron networks for text analysis
- Each word associated with neighboring context words
- Predictive model uses errors to update word representations and network settings

**Functioning:**
- Lightweight version of neural networks: few layers, quick training
- Focuses on context representation, not abstract meta-relations
- Probabilistic model for selecting random words in corpus

**Developments:**
- GLOVE uses co-occurrences to reproduce Word2Vec results
- Recent methods like FastText moving away from neural networks

**Applications:**
- Identify main groups of entities and links within text
- Uncover implicit classifications and ontologies in various productions

**Benefits:**
- Flexible data models for exploring relationships between words.

**Word2Vec vs Topics:**
- Topics lock you into preconceived ideas about topics
- Word2Vec allows for open exploration of relations embedded in words.

**History:**
- Originally designed with search engine perspective in mind
- Used to reconstruct discourse progress and identify subgenres.

**Technical Details:**
- Predictive model based on positive affinity between terms (0.7 or above)
- Connects related pairs of words, forming networks for specific discourses.

### Test of release of semantic sets

**Semantic Ensemble Extraction Using Word2Vec**

**Approach**:
- Identify relevant semantic sets by:
  - Using a sufficiently high context (10 words)
  - Upstream syntax labeling using Talisman application
  - Cutting Word2Vec on lemmatized corpus to obtain "reference" forms
  - Limiting unwanted cross-over terms

**Process**:
1. **Matrix of Affinities**:
   - Calculate cosine distance between words using R and "cosine" function in "lsa" extension
   - Obtain a big reciprocal table with 1805 entries and as many columns
   - Limit the size by selecting terms appearing more than five times

2. **Project Main Relationships into a Network**:
   - Use "melt" function in "reshape2" extension to create network graph with source, target, and relationship value
   - Activate recognition of clusters in Gephi to visualize the semantic ensembles

**Findings**:
- **Opera Record**: Divided into characters and places/objects/elements of intrigues
  - Corroborates intuition that semantic ensembles overlap with stylistic sets
- **"Feelings"**: Primacy accorded to qualifiers with musical connotation
  - Not feelings in general, but "feelings in the perspective of a description of musical reception around mid-19th century"
- **Musical Description**: Divided into formal musical description and musical performance

**Additional Findings (Zoom):**
- **Feelings**: Subdivided into feelings directly accounting for musical idea and general romantic phraseology

### Some perspectives

**Word Vectors as a Measure for Text Qualification**

**Benefits of Word Vectors:**
- Allow for efficient transmission of information about large corpora
- Occupy minimal storage space

**Current Challenges:**
1. **Comparing vectors of distinct words**:
   - To compare results across different textual productions (e.g., literature vs. press)
   - Methods like "Procruste Orthogonal" enable recrossing word embeddings from distinct corpus
2. **Resolving ambiguity of terms:**
   - Word embedding models work on raw words, no external resources for homonyms/polysemias
   - Techniques like "rejection of the vector" can help resolve ambiguity in specific contexts
3. **Automating the process**:
   - Manual identification of "families" of clusters requires human judgment
   - Comparing terms with ontologies or Wikidata for common names, personal names resolution
4. **Openness of neural network methods:**
   - Initiatives to make selection methods more accessible (e.g., interactive visualizations)
   - Analyzing dimensions in Word2Vec models reveals potential themes and clusters
5. **Implications for text analysis**:
   - Word vectors can be useful in poorly studied corpuses, such as newspapers and manuscripts
   - Automated tools will contribute to large-scale irrigation of the web with text data.

