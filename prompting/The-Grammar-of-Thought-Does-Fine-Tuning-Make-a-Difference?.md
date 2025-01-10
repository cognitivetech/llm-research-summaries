# The Grammar of Thought Does Fine-Tuning Make a Difference?

source: https://peytoncasper.com/blog/tone-evaluation/index.html
Peyton Casper
November 22, 2024

## I: Charting the Unseen

**Stylometry: Analyzing Literary Styles Through Data Science**

**Introduction:**
- Field of study: stylometry, statistical analysis of literary style
- Guided by "Stylometry Methods in Python" resource

**Authors and Their Works:**
1. J.K. Rowling: Harry Potter series
2. Tade Thompson: The Wormwood Trilogy (science fiction)
3. Andre Agassi: Autobiography

**Initial Findings:**
- Similarities in surface-level analysis using Mendenhall's Characteristic Curves of Composition
- All authors share language building blocks

**Deep Dive:**
1. Word length distribution analysis
2. Word frequency comparison within and across authors
3. Employing Jensen-Shannon divergence for quantitative analysis
4. Lower values indicate closer alignment, higher values signify greater divergence in writing styles.
5. Captures subtle shifts in word choice and pattern unique to each author's voice.

### Jensen-Shannon Heatmap

**Heatmap Analysis:**
- Clustering of works by same author (HP1 and HP2, WT1 and WT2, AA) due to consistent narrative voices
- Lower divergence values between their texts

**Authorship Identification:**
- Jensen-Shannon divergence not definitive for authorship identification
- Punctuation analysis complements this method

**Comparison of Authors' Writing Styles:**
- **J.K. Rowling**:
  * Use of quotation marks, single quotes, and apostrophes in dialogue-rich storytelling
  * Consistent with her character interaction focused series (Harry Potter)
- **Tade Thompson and Andre Agassi**:
  * Favor measured use of punctuation with steady rhythm of periods and commas.

## II: Data Alchemy

**Storytelling Mechanics: Narrative Elements and Author's Style**

**Narrative Components:**
- **Action**: Weaving seamlessly into dialogue (J.K. Rowling)
- **Dialogue**: Richly detailed yet fast-moving scenes (J.K. Rowling)
- **Exposition**: Carefully crafted context in dynamic worlds (Tade Thompson)
- **Description**: Vivid and grounded in action or thought (Authors)
- **Inner Thoughts**: Intimate first-person accounts, emotional landscapes (Andre Agassi)

**Model Experiments:**
- Dataset division into paragraphs
- Generating summaries using GPT-3.5 for structured pairs
- Fine-tuning experiments to explore dataset size impact on model performance
- Identifying sweet spot where flexibility and overfitting balance (Authors)

**Experiment Insights:**
- J.K. Rowling: Equilibrium of action and dialogue
- Tade Thompson: Heavily weighted towards action and exposition
- Andre Agassi: Focus on inner thoughts and exposition in first-person account

**Classification System:**
- Aspect Enum: Dialogue, Action, Exposition, Description, Inner Thoughts (Python)
- ParagraphClassification TypedDict: paragraph: str, aspect: str
- GenerativeModel "gemini-1.5-pro-latest" for classification using enum output.

## Capturing the Magic of J.K. Rowling's Style

**GPT-4 Embodies J.K. Rowling's Style**

As GPT-4 fine-tunes, its radar chart evolves across five dimensions, reflecting its growing similarity to Rowling's writing style (brown). The model's voice (orange) develops over time, as shown by an interactive slider on the right, allowing you to explore this transformation visually.

## III: Drawing the Curtain

**Exploring Stylistic Evolution in Writing:**
* **Fine-tuning and prompt engineering**: contributors to model's stylistic evolution (interactive slider for adjusting emphasis on inner thoughts)
* Rowling's unique blend of action and introspection: fine-tuning brings the model closer
* Traditional stylometry + language models: potential framework for taming models

**Discoveries from Exploration:**
* Fine-tuning vs. large training sets: smaller, focused sets often outperform larger ones
* Quality and precision of training data eclipses sheer quantity in data science
* Writing style transcends word choice or sentence structure: intricate choreography of narrative elements
* Masters balance action, dialogue, description with unique rhythms

**Insights on Data Extraction:**
* Quantifying aspects of style: replicable but not the true alchemy
* Writers' voices shaped by memories and experiences: defies precise measurement
* Infinite question: how do writers with an "inner voice" differ from those without one?

## Where to Take It from Here

"I'll share three promising areas for exploring AI research that I plan to delve into soon."

### Synthetic Dataset Generation

Fine-tune models by steering towards specific narrative aspects like 80% dialogue and 20% description to assess whether fine-tuning can precisely control chosen styles.

* Use GPT-4 for targeted training data
* Fine-tune on extreme distributions (e.g. 80/20 dialogue/description)
* Measure style transfer precision

### Embeddings vs. Stylometry

Compare transformer embeddings to traditional stylometric measures to see if they capture similar writing style attributes.

1. Generate text embeddings.
2. Compare with stylometric features.
3. Identify which embedding dimensions correspond to specific attributes.

### Statistical Validation
Run stylometric analysis on human and LLM evaluations of style transfer quality to validate statistical approaches for capturing subjective writing style assessment. Generate large outputs, run full analysis, and compare results.


