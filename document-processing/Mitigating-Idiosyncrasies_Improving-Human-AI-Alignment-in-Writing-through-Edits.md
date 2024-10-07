# Can AI writing be salvaged? Mitigating Idiosyncrasies and Improving Human-AI Alignment in the Writing Process through Edits

by TUHIN CHAKRABARTY, PHILIPPE LABAN, CHIEN-SHENG WU,  Salesforce AI Research, USA
https://arxiv.org/pdf/2409.14509

## Abstract
**Background:**
- LLM-based applications are being used to help people write
- LLM-generated text is appearing on social media, journalism, and in classrooms
- Differences between human-written and LLM-generated text are unclear

**Findings from Professional Writers:**
- Agreed on undesirable idiosyncrasies in LLM-generated text
- Categorized the idiosyncrasies into 7 categories: clichés, unnecessary exposition, etc.

**LAMP Corpus:**
- Collected 1,057 LLM-generated paragraphs edited by professional writers according to taxonomy
- Analysis of LAMP revealed no significant difference in writing quality among LLMs used (GPT4o, Claude-3.5-Sonnet, Llama-3.1-70b)
- Common limitations found across model families

**Automatic Editing Methods:**
- Large-scale preference annotation showed promise in improving alignment between LLM-generated and human-written text
- Experts prefer text edited by other experts, but automatic editing methods show potential.

**Contributions:**
- Human-centered computing: empirical studies in HCI, collaborative and social computing
- Computing methodologies: natural language generation

**Additional Keywords:**
- Human-AI collaboration
- Large Language Models
- Design Methods
- Text Editing
- Natural Language Generation
- Evaluation
- Writing Assistance
- Generative AI
- Homogenization
- Alignment
- Behavioral Science.

## 1 Introduction

**Introduction:**
- AI has potential to revolutionize writing
- LLMs shown to assist with various writing tasks
- Aligning LLMs with human preferences enhances tool usability

**Challenges:**
- Homogenization of content and linguistic diversity
- Reduction in content diversity leads to monoculture
- Current AI-assisted writing tools improve quality, but introduce challenges
  *Pairing outputs for preference ranking can be flawed*

**Proposed Solution: Alignment via Edits**
- Humans edit responses to enhance alignment
- Edited responses contain fewer undesirable traits
- Paragraph-level editing balances granularity and scope

**Methodology:**
1. Create comprehensive taxonomy of edit categories based on expert writing practices
2. Recruit 18 writers to edit LLM-generated text using the taxonomy
3. Focus on literary fiction and creative non-fiction genres
4. Conduct thorough analysis of editing process, differences between model families, stylistic idiosyncrasies
5. Investigate if LLMs can automatically detect and rewrite their own idiosyncrasies
6. Discuss future improvements for co-writing experience

**Contributions:**
1. Proposed comprehensive edit taxonomy informed by expert writing practices
2. Released LAMP corpus containing 1057 <instructions, response> pairs edited by professional writers
3. Conducted thorough analysis of editing process, differences between model families, stylistic idiosyncrasies
4. Statistically significant results showing preference trend for LLM-generated text edits
5. Discussed future improvements for co-writing experience.

## 2 RELATED WORK 
### 2.1 Text Editing in HCI 

**Related Work: Text Editing and NLP**

**Text Editing in HCI**:
- Research on improving digital writing tools' efficiency and usability
- Systems like Soylent, MicroWriter, and WearWrite focused on task breakdown, cost management, and minimizing delays
- Robertson and Black proposed a goal-fate analysis model for text editing behavior

**Text Editing in NLP**:
- Various text editing tasks explored in NLP research
- Advancement of Large Language Models (LLMs) enabled AI-assisted writing tools
- Faltings et al. released WikiDocEdits dataset, proposed interactive text generation setting
- Raheja et al. proposed instruction-based editing system using fine-tuned language models
- Shu et al. developed strategies for cross-sentence rewriting and introduced OpenRewriteEval benchmark
- Reid and Neubig modeled multi-step editing processes to mimic human content creation
- Kim et al. presented a system that iteratively improves fluency, clarity, coherence, and style by detecting editable spans
- Yang et al. developed taxonomy and classifier for Wikipedia edit intentions

**Issues in AI Writing**:
- Chakrabarty et al., Ippolito et al., Marco et al., Mirowski et al. showed LLM-generated text lacks nuance, subtext, and rhetorical complexity
- Recent work from Mirowski et al. demonstrated LLMs fail to act as creativity support tools for comedy writing
- Subbiah et al. showed LLMs struggle with specificity and interpretation of difficult subtext
- Tian et al. shed light on how LLM-generated stories are homogeneously positive and lack tension
- Lee et al., Li et al. revealed users benefit from AI assistance in productivity and confidence but face potential drawbacks
- Hohenstein and Jung found LLM-generated text suggestions can affect human writer's emotional tone

**Human-AI Alignment in Writing**:
- AI tools have transformed writing processes, creating new criteria for future AI writing assistants
- Users benefit from AI assistance in productivity and confidence but face potential drawbacks like decreased accountability and diversity in writing
- LLMs used in writing assistance can significantly influence human-authored content
- Recent work demonstrated language models can enhance outputs via feedback but may result in reward hacking during alignment training.

## 3 DESIGN CONSIDERATIONS TO IMPROVE AI WRITING

**Design Considerations to Improve AI Writing**

**Principle 1: Develop a Comprehensive Edit Taxonomy Grounded in Expert Writing Practices**:
- Emphasizes creating a comprehensive taxonomy of edit categories
- Rooted in expert editor and writer's practices
- Aims to provide an approach for analyzing and enhancing LLM-generated text
- Allows for granular understanding of specific areas where AI writing may fall short
- Enables targeted improvements by enabling identification of specific editing needs

**Principle 2: Leverage Edits to Balance Meaning Preservation and Substantive Semantic Changes**:
- Preserving core meaning and intent is crucial to maintain coherence and faithfulness
- Introducing substantive semantic changes is often required for good writing standards
- Prior work focused on low-level syntactic operations or semantic edits
- LLM-generated text benefits from both meaning preserving and changing edits
- Aim to navigate the tension between maintaining original meaning and introducing improvements

**Principle 3: Utilize Edits as a Mechanism for Enhancing Human-AI Alignment in Writing**:
- Current AI writing systems use pre-trained language models (LMs)
- Reinforcement learning from human feedback (RLHF) is used to guide LMs towards desired outcomes
- Common feedback is binary preferences between pairs of examples from one or more LLMs
- Edits as a mechanism for enhancing alignment can improve preference data collection
- This method has been adopted by contemporaneous work from Meta AI and Contextual AI

## 4 LARGE SCALE DATA COLLECTION PROCESS

**Approach for Creating a Valid Test-Bed to Evaluate LLM-Generated Text**
- Three-step approach: (1) Select original human-written paragraphs, (2) Reverse-engineer each into writing instructions, and (3) Prompt several LLMs to generate responses based on these instructions.

**Step 1: Curating Source Material**
- Manually extract 100-700 pieces of writing from five well-regarded publication venues
- Isolate individual paragraphs and manually review for coherence, yielding approximately 1200
- Largest representation is Literary Fiction (80%)

**Step 2: Instruction Generation**
- Use Instruction Backtranslation to automatically generate open-ended questions based on each selected paragraph
- Manually verify and filter generated instructions, yielding a total of 1,057 writing instructions
- Ensure that each LLM responds to instruction across all domains in equal proportion

**Step 3: Prompting LLMs for Responses**
- Use three state-of-the-art LLMs (GPT-4o, Claude-3.5-Sonnet, Llama 3.1-70b) to generate responses based on instructions
- Provide LLMs with writing instruction, genre, and source to ensure high-quality responses
- Obtain 1,057 writing instruction, response pairs averaging 205 words

**Formative Study**:
- Foundation for three studies: formative study, full-scale editing annotation, preference annotation.

### 4.2 Formative study: formulating the taxonomy for fine-grained edits

**Formative Study Methodology**
- **Participants**: observed writers with copy-editing experience in creative writing domain
- **Phases**: 1) objectives explained via video conference; 2) participants annotated problematic response spans, suggested rewrites, and tagged issues with free-form categories on web application
- **Selection of Participants**: limited to individuals with MFA in Creative Writing

**Participant Demographics**
| Participant | Gender | Age | Educational Background | Genre Specialization |
|---|---|---|---|---|
| W1, W2, W3, W5, W6, W7 | Male | 28, 29, 31, 30, 35 | MFA in Creative Writing (Fiction/Poetry) | N/A |
| W4 | Female | 30 | MFA in Non-Fiction | N/A |
| W8 | Male | 26 | MFA in Fiction | N/A |

**Study Process and Results**
- Participants annotated problems, suggested rewrites, and tagged issues on web application (editing a response took 4-6 minutes)
- In total, eight participants edited 200 paragraphs, annotating roughly 1,600 edits attributed to 50 distinct initial categories.

**Future Analysis**: using data as foundation for developing taxonomy for categorizing edits.

### 4.3 From initial to final categorization of edits

**Semantic Overlap and Taxonomy Consolidation:**
- Significant overlap among initial 50 categories used by participants
- Corresponding categories: "Show don't tell" (W4) and "Unnecessary because implied" (W6)
- General inductive approach for qualitative data analysis used to synthesize initial categories into a comprehensive taxonomy

**Low-Level Grouping:**
- Two authors independently bucketed categories into low-level groups
- Iterative discussions refined groups, reducing overlap and establishing shared groupings

**High-Level Categorization:**
- Low-level categories aggregated into high-level categories
- Each high-level category named to reflect its generalized representation

**Final Taxonomy:**
1. **Clarity**: Edits that improve clarity by addressing ambiguous statements or expressions.
   - Example: "Please clarify the relationship between A and B." (W2)
2. **Conciseness**: Edits that remove unnecessary words, phrases, or sentences without losing meaning.
   - Example: "Delete redundant adjectives." (W13)
3. **Grammar**: Correcting grammatical errors such as tense inconsistencies and subject-verb agreements.
   - Example: "Revise sentence for correct verb tense." (W8)
4. **Consistency**: Edits that maintain consistency in terms of style, format, or tone throughout the text.
   - Example: "Maintain consistent heading formatting." (W19)
5. **Coherence**: Correcting inconsistencies and ensuring logical flow between ideas within a paragraph or document.
   - Example: "Check for internal consistency in argument." (W20)
6. **Terminology**: Ensuring accurate use of terms throughout the text, especially when technical jargon is involved.
   - Example: "Revise sentence to use proper terminology." (W15)
7. **Contextual Relevance**: Edits that address irrelevant information or ensure that all content remains relevant to the intended audience.
   - Example: "Remove non-essential information." (W23)

### 4.4 Final Taxonomy for Fine-Grained Edits

**Final Taxonomy for Fine-Grained Edits (Category: Cliché)**
* Overused phrases, ideas, or sentences that lose original meaning
* Characterized by vivid analogies or exaggerations from everyday life
* Frequent use in writing viewed as sign of inexperience or lack of originality [35]
* Replacing clichés with fresh language improves writing and engages readers more effectively.

**Final Taxonomy for Fine-Grained Edits (Category: Unnecessary/Redundant Exposition)**
* Excessive, repetitive, or implied information in writing
* Restating the obvious or providing details of little value
* Embraces "show, don't tell" principle to allow readers to infer meaning from context [11,17,48,73]
* Effective writing allows core message to shine through without unnecessary verbiage.

**Final Taxonomy for Fine-Grained Edits (Category: Purple Prose)**
* Excessively elaborate writing that disrupts narrative flow
* Often difficult to read due to sprawling sentences, abstract words, excessive adjectives, and metaphors [1]
* Careful editing can trim purple prose by replacing ornate language with more direct expressions
* Results in clearer writing that preserves author’s voice.

**Final Taxonomy for Fine-Grained Edits (Category: Poor Sentence Structure)**
* Reduces clarity and readability of writing [8,49,67]
* A lack of proper transitions can make text feel disjointed and hard to follow
* Editing for clarity often reveals it’s better to split convoluted thought into two sentences
* Overwhelming readers with long and complex sentences [15]
* Careful editing leads to more coherent and fluent text.

**Final Taxonomy for Fine-Grained Edits (Category: Lack of Specificity and Detail)**
* Broad generalizations that fail to engage readers
* Lack of vivid details leaves readers unable to visualize scenes or connect with writing on a deeper level
* Good writing focuses on adding details, contextualizing information, and deepening internality [26,34,52]
* Developing unique voice through carefully chosen words and phrases can inject personality into the text.

**Final Taxonomy for Fine-Grained Edits (Category: Awkward Word Choice and Phrasing)**
* Can significantly reduce writing quality and disengage readers
* Misused or disproportionate use of certain words, unclear pronoun references, overuse of passive voice [51]
* Editing plays a crucial role in refining elements and replacing imprecise or ill-fitting terms with appropriate alternatives.

**Final Taxonomy for Fine-Grained Edits (Category: Tense Inconsistency)**
* Prevalent issue in writing that detracts from overall coherence [70]
* Occurs when a writer shifts between past, present, and future tenses within the same paragraph or sentence
* Careful editing plays a crucial role in addressing inconsistencies by paying close attention to verb forms and temporal indicators.

### 4.5 Collecting Edits on LLM Generated responses

**Collecting Edits on LLM Generated Responses**
- Conducted a larger-scale annotation study to collect edits from writers on LLM-generated responses
- Categorized edits using the established taxonomy
- Participants provided access to an editing interface with instructions and LLM-generated responses
- Participants could select any span of text in the response and suggest a rewrite, choosing from 7 predefined categories in the taxonomy
- Participants received training on the taxonomy via email, including example edits for each category
- No set limit on edits per response, but participants were urged to improve the text as they saw fit
- Interface logged all edits chronologically and offered an undo feature to track entire editing process
- Participants assigned Initial Writing Quality Score (IWQS) and Final Writing Quality Score (FWQS) for each sample
- Self-reported writing quality scoring system served as a signal for writers to recognize improvements, set personal goals, and develop intrinsic motivation
- Participants included writers, editors, teachers, poets, translators, journalists, directors, and screenwriters with formal creative writing backgrounds (Table 4)
- Writers completed the task in batches of 25 responses over 2.5 months, taking 3 hours per batch, and were compensated $100 USD for each batch
- Recruited 18 writers to edit LLM-generated responses, with some participating from the formative study
- A total of 1,057 <instruction, response> pairs were edited by at least one participant, and 50 responses were edited by three participants

## 5 The LAMP Corpus 5.1 Overall Statistics

**LAMP Corpus Analysis: Professional Editing of LLM-Generated Text**

**Overall Statistics:**
- Collaborated with 18 writers to edit 1,057 LLM-generated paragraphs
- Totaled approximately 8,035 fine-grained edits
- Includes data from Claude3.5 Sonnet (368), GPT4o (393), and Llama3.1-70B (296) models

**Editing Operations:**
- Replacements: most frequent at 74%
- Deletions: second most common with 18% of all edits
- Insertions: least common, accounting for only 8% of all edits
- Variability in editing styles among participants

**Analyzing Editing Operations:**
- Figure 4a shows edit operations by participant for each paragraph
- Replacements are most frequent due to correcting errors or improving clarity
- Deletions often eliminate unnecessary information or redundant sentences
- Insertions used less frequently, mainly to add specific details or improve flow

**Semantic Similarity Analysis:**
- To quantify meaning-preserving vs. meaning-changing edits, calculate semantic similarity using BERT score (threshold of 0.67)
- Of 6468 non-deletion edits, 70% are meaning-preserving and the rest meaning-changing
- Supports Design Principle 2: Allows participants to provide Initial and Final Writing Quality Scores (IWQS and FWQS) for each paragraph

**Distribution of Writing Quality Scores:**
- Figure 4b shows distribution of writing quality scores for each participant, revealing significant variability
- Calibration of writing quality scores is a known challenge, but normalized using mean subtraction, division by standard deviation, and re-scaling to the 1 to 10 range
- Subsequent analyses use these normalized scores

**Edit Distance:**
- Calculate edit distance between original LLM-generated text and final edited text using character-level Levenshtein distance
- Negative correlation between edit distance and IWQS (Pearson's r=−0.31), indicating less editing required for higher perceived quality texts

**Writing Quality Comparison:**
- No significant difference in writing quality across GPT-4o, Claude 3.5 Sonnet, and Llama3.1-70B models in creative non-fiction (average IWQS of 5.2 for the first two, slightly better than Llama3.1-70B with an average IWQS of 5.0)
- All models show a slight decrease in performance for fictional instructions (average IWQS of 4.5)

**Edit Categories:**
- Figure 5a shows edit categories applied by writers to texts from three LLMs: Awkward Word Choice and Phrasing (28%), Poor Sentence Structure (20%), Unnecessary/Redundant Exposition (18%), Clichés (17%)
- Minute differences include GPT-4o using more purple prose and Llama3.1-70B generating more unnecessary exposition
- Overall, LLMs exhibit similar idiosyncrasies that are edited out in proportions by professional writers

**Relationship Between Editing Categories and IWQS:**
- Higher IWQS scores correspond to fewer total edits (Figure 5b)
- Unnecessary/Redundant Exposition and Lack of Specificity and Detail remain relatively constant, while Awkward Word Choice and Phrasing, as well as Cliché, decrease with increasing IWQS

### 5.2 Writers differ greatly in the amount of editing they do : But to what extent?

**Differences in Writing Approaches**

**Writers' Editing Styles:**
- Vary based on personal or organizational philosophy
- Prioritize preserving original voice vs. intervening
- Make fewer but more impactful changes vs. numerous small revisions

**Editing Differences among Writers (W3, W12, W16):**
- W3: 9.4 edits on average
- W12 and W16: 6.0 and 6.3 edits on average
- Span level precision between the writers was 0.57, suggesting moderate agreement

**Disagreements in Categorization:**
| Writer | Selected Spans | Assigned Categories |
|---|---|---|
| W3 and W16 | " ,the numbers glaring back at me like an unsolvable riddle" | Unnecessary/Redundant Exposition (Cliche) vs. Purple Prose |
| W3 and W12 | "and an unsettling sense of mystery that gnawed at me more than the inexplicable weight itself" | Cliché vs. None (did not edit this span)

**Implications:**
- Different interpretations can be correct, depending on usage and context
- Overused phrases or clichés may result in redundant exposition that doesn't add value to the narrative.

### 5.3 Are there any specific stylistic idiosyncrasies in LLM generated responses?

**LLM Generated Responses: Stylistic Idiosyncrasies**

**Syntactic Patterns in LLM Responses**:
- DT NN IN NN CC (54% of times, edited): "a mix of pride and, a mix of fear and, a sense of protection and, a sense of wonder and, a means of connection and, a pang of nostalgia and, a pang of disappointment"
- DT JJ NN IN PRP$ (40% of times, edited): "a constant reminder of his, the mundane routine of our, the intricate tapestry of its, the subtle shift in their, the potential weight of its"
- DT NN IN JJ NN (27% of times, edited): "the fabric of daily life, a moment of genuine connection, a life of absolute relaxation, the face of inevitable loss"
- IN DT NN IN NN CC (45% of times, edited): "with a mix of wariness and, by the hum of traffic and, in a flurry of pursuit and, into a world of precision and, in a gesture of comfort and, in a storm of pain and"

**Syntactic Patterns Captured by Shaib et al.'s Research**:
- **Recent work from Shaib et al.**: Syntactic patterns with Part-of-speech8as abstract representations of texts can capture more subtle repetitions than mere text memorization. Language models tend to use repetitive syntactic templates more often than humans, and these patterns can help evaluate style memorization in language models.
- **Table 1**: The researchers looked at the 50 most common templates in LLM-generated responses and found that 15 templates do not occur as frequently in original human-written seed paragraphs.

**Idiosyncrasies Found in LLM-Generated Responses**:
- Table 8 shows representative sequences corresponding to particular syntactic patterns present in higher proportion in LLM-generated responses. These sequences constitute categories of **Clichés, Unnecessary/Redundant Exposition**, or **Poor Sentence Structure**.
- **Awkward Words/Phrases Occurring Disproportionately**: For instance, the word "unspoken" occurs in about 15% of LLM-generated responses. Phrases such as "weight of," "sense of," and "mix of" occur very rarely or not at all in original seed paragraphs, while they occur frequently in LLM-generated responses.
- **Peuliar and Uncommon Phrases Generated by LLMs**: The most surprising finding is that all 3 LLMs generate these idiosyncratic words/phrases, suggesting possible overlap/mixture in instruction tuning data across model families or one model trained on synthetic data generated from another.

**Figure 6: Distribution of Peculiar and Odd Words and Phrases**: The distribution of peculiar and odd words and phrases occurring in LLM-generated text vs. human-written text in the LAMP Corpus is shown.

## 6 AUTOMATIC DETECTION AND REWRITING OF LLM IDIOSYNCRASIES

**Automatic Detection and Rewriting of LLM Idiosyncrasies**

**Expert Text Editing**:
- Can analyze and reduce LLM idiosyncrasies at a small scale
- Automated methods needed for resolution at a larger scale

**Developing Techniques**:
- Building on Hayes et al. [41] and Scardamalia [92]
- Separate detection and rewriting tasks

**Evaluation**:
- Using LAMP Corpus annotations
- Conducted preference annotation study with LAMP Corpus writers
  - Comparing human and LLM-produced edits

**Data Splitting**:
- 146 of 1057 LAMP Corpus paragraphs for training
- Remaining paragraphs for testing

### 6.1 Automatic detection of problematic spans in LLM-generated text

**Automatic Detection of Problematic Spans in LLM-Generated Text**

**Problem Formulation**:
- Detecting problematic spans in LLM-generated text as a multi-span categorical extraction problem
- Given a paragraph, output non-overlapping spans with categories from the LAMP Corpus

**Evaluation Metrics**:
- **Span-level precision**: Measures degree of overlap between predicted and reference spans
- Precision scores indicate model's ability to identify correct boundaries without over-predicting
- Implemented as **General Precision** (credits span selection) and **Categorical Precision** (requires correct category assignment)

**Automated Method Results**:
- Llama3.1-70b, GPT-4o, Claude3.5-Sonnet: General Precision 0.42–0.46, Categorical Precision 0.14–0.18
- Improvement from 2-shot to 5-shot prompts but plateaus afterwards
- Expert agreement (50 edited paragraphs): General Precision 0.57, Categorical Precision 0.23

**Limitations**:
- Performance improvement possible
- Category agreement may differ even when problematic spans are identified

**Example of Span Comparison**:
- [Appendix A.2] provides a simplified example for computing General and Categorical Precision

### 6.2 Automatic rewriting of problematic spans in any LLM-generated text

**Automatic Rewriting of Problematic Spans using LLMs**
* Use few-shot prompting [10] for LLMs to propose improvements
* Design prompts for each edit category with examples from the LAMP Corpus and target span
* Prompts include definition, 25 examples, expert rewrite, and input paragraph
* Examples of problematic spans identified by a writer and an LLM in Table 10

**Two-step Pipeline for Editing Paragraphs:**
1. Detection identifies problematic spans and assigns categories
2. Rewriting uses category-specific prompts to revise each detected span
3. Replace all problematic spans with their rewrites in the original paragraph

**Manual Evaluation:**
* Judge complete pipeline that edits an entire paragraph through manual evaluation by 12 writers on the LAMP Corpus
* Describe manual experiment next.

### 6.3 Evaluating Automatic Editing of LLM-generated Text

**Evaluating Automatic Editing of LLM-generated Text**

**Background:**
- Experts rank three variants of a paragraph: unedited LLM-generated, Writer-edited version, and LLM-edited version using pipeline to detect and rewrite problematic spans.
- Two sub-conditions for LLM-edited: Writer Detected (pipeline relies on reference spans selected by the writer) and LLM Predicted (automatic detection of problematic spans).

**Evaluation Task:**
1. Participants read three variants of a paragraph and rank them in terms of overall preference.
2. Included all four conditions initially but redesigned task to have participants judge only three conditions per annotation session.
3. Anonymized paragraphs and did not inform participants about the difference between them.

**Results:**
- LLM-edited > LLM-generated (p-value: 1.3e-11 for Writer Predicted spans; 2.8e-13 for LLM Predicted spans) and Writer-edited > LLM-generated (p-value: 1.1e-26 for Writer Predicted spans; 1.17e-31 for LLM Predicted spans) using Wilcoxon signed-rank test.
- Overall agreement of 0.505 among annotators, suggesting moderate level of agreement.
- Preference results: Writer-edited most preferred (65% of the time, average rank 1.5), followed by LLM-edited variants (average rank 1.99 for both conditions), and original LLM-generated paragraphs least preferred (60% of the time, average rank 2.51).

**Implications:**
- Automatic editing can improve writing quality to match that of professional writers.
- Rewriting module is crucial in automated text editing pipelines as it dictates overall performance.

## 7 DISCUSSION
### 7.1 How is editing human writing different from LLM-generated text?

**Editing Human Writing vs. LLM-Generated Text**

**Challenges**:
- Editing human writing: preserving unique voice, style, and context
- LLM-generated text: removing unusual metaphors, nonsensical descriptions, improving tone, consistency

**Differences**:
- Human writing: nuanced expressions, personal style, contextual references
- LLM-generated text: lacks consistent tone, repetitive patterns

**Editor's Feedback**:
- More extensive editing needed for LLM text
- Removing clichés, histrionic descriptions, and exposition in LLM text
- Need to improve overall tone (avoid impersonal and mechanical writing)
- Volume of changes required for LLM text is higher compared to human writing
- Repetitive nature of LLM content can make the editor feel robotic while editing it.

### 7.2 How well can LLMs mimic edits from writers?

**LLMs' Ability to Mimic Writers' Edits**

**Question:** Can LLMs accurately mimic edits made by writers?

**Findings**:
- Automatically edited paragraphs often rank second or first in preference ranking results (Section 6.3)
- Raising questions about LLM's ability to analyze textual patterns and generate content similar to a given writer's edit

**Examples of Edits:**
1. Cliché Removal:
   - Original: "Janet lay in bed each night, her mind a whirlpool of restless thoughts"
   - LLM Edit: "Janet lay in bed each night, unable to sleep"
   - Writer's Edit: "Each night, Janet lay prone in her bed and unable to sleep"
2. Splitting Run-on Sentences:
   - Original: "Sarah froze, realizing it was her high school sweetheart, Alex, whom she hadn’t seen in over a decade"
   - LLM Edit: "Sarah froze. It was Alex, her high school sweetheart. She hadn’t seen him in over a decade"
3. Cliché Replacement:
   - Original: "Her irritation slowly morphed into a strange, disconnected calm"
   - Writer's Edit: "After all, the noise just meant that she wasn’t the only one awake at this hour."
   - LLM Edit: "Her irritation slowly morphed into a strange, disconnected calm. The repetitive thump-thump-thump became almost hypnotic, lulling her into a trance-like state"

**Limitations**:
- Reliance on few-shot instructions may limit model's learning ability
- Training on larger datasets like LAMP Corpus or more data could potentially improve edit quality.

### 7.3 What recommendations can we provide for future LLM-based writing support tools that aspire to improve co-writing experience?

**Recommendations for Future LLM-Based Writing Support Tools**

**Improving Co-Writing Experience**:
- LLMs can produce grammatically correct sentences free of spelling errors, but require extensive learning to effectively assist humans in improving writing
- Orwell's rule: "Never use a long word when a short one will do"
- LLMs overuse lofty words and clichés due to the technology behind them
- **Clichés** become very likely for LLMs because they appear so often
- LLMs need to learn how to identify and write without clichés to make writing engaging for every reader
- **Overwriting** is a bigger problem than underwriting
- **Writing by omission**, as Pulitzer Prize-winning writer John McPhee calls it, is the process of avoiding unnecessary exposition
- Good writing hangs on **structure**, and long, run-on sentences are hard to read. LLMs need to know when and how to split effectively to manage flow and clarity.

### 7.4 What are the potential long-term effects on language evolution and writing styles as LLM becomes more prevalent and how can aligned editing tools

**Potential Long-Term Effects of LLMs on Language Evolution and Writing Styles**

**Impact of Large Language Models (LLMs)**:
- Could significantly impact language evolution and writing styles over time
- Increasing prevalence could lead to more homogenized writing as people rely on LLM-generated content
- Possible reduction in linguistic diversity and individual voice

**Role of Editing Tools**:
- Well-designed editing tools aligned with expert writing practices could help counteract these effects
- Encourage nuanced and sophisticated language use, preserve stylistic diversity, and promote critical thinking about word choice and sentence structure
- Highlight elements of expert writing to elevate overall writing quality while allowing for personal expression
- Potentially steer language evolution towards greater clarity, precision, and effectiveness in communication.

## 8 Limitations

**Limitations of Study**
* Limited generalizability due to small participant pool (18 MFA-trained creative writers)
* Findings may not apply to other genres like technical writing, journalism, or scientific writing
* Selected LLMs are among most advanced but do not represent entire spectrum of AI writing abilities
* Evaluation of writing quality is subjective, with potential for expert disagreement on improvements
* Automated methods for detecting and rewriting problematic spans relied on few-shot learning with limited examples
* Paragraph-level editing may miss broader structural or thematic issues apparent only in longer pieces of writing
* Professional writers editing AI-generated text for monetary compensation versus editing one's own work
* Fatigue from repetitive task of editing multiple AI-generated paragraphs could lead to less thorough or thoughtful edits.

## 9 Conclusion

**Key Findings**

**Taxonomy of Edit Categories**:
- Developed a taxonomy of edit categories grounded in established writing practices

**LAMP Corpus**:
- Created the LAMP corpus containing over 8,000 fine-grained edits by professional writers on LLM-generated text

**Automatic Detection and Rewriting**:
- Designed methods for automatic detection and rewriting of problematic spans in LLM generated text
- Found that **automated methods using few-shot prompting are able to detect and rewrite problematic spans**, though not matching human expert performance

**Preference Evaluations**:
- **Writers consistently rank text edited by other writers highest**
- Followed by LLM-edited text, with unedited LLM-generated text ranking lowest

**Conclusion**:
- As AI text generation becomes more prevalent, developing robust editing and alignment techniques will be crucial to ensure AI systems produce high-quality writing that meets human standards and enhances creativity and linguistic diversity.

## A Appendix

**Prompt Generation: Instruction Summarization**

**Paragraph Summary**:
- Summarize this paragraph into a single sentence open-ended question
- Response prompt: Imagine you are a fiction writer for the NewYorker/Modern Love section, etc. Write a response following instructions to be original and focus on nuance, simplicity, and subtext

**Instructions for Writing Prompts**:
1. Be a fiction writer for specific sections (NewYorker, Modern Love)
2. Avoid clichés or overused tropes
3. Use simple language with a focus on nuance and subtext
4. Start directly with your response
5. For NY Times Cooking: Write 10-15 sentences in response to a question
6. For NY Times Travel: Write 10-15 sentences in response to a question
7. As a beloved female Internet advice columnist, be deeply felt and frank with personal experience
8. Be original, avoid clichés or overused tropes, focus on nuance, simplicity and subtext
9. Start directly with your response

**Table 13**: Prompts for generating instructions and responses

### A.1 Idiosyncracy Span Detection Prompt

**Idiosyncrasy Span Detection Prompt:**
* Provide feedback on problematic spans of text in a paragraph by selecting them and assigning error categories
* Seven error categories: Awkward Word Choice and Phrasing, Cliche, Poor Sentence Structure, Unnecessary/Redundant Exposition, Lack of Specificity and Detail, Purple Prose, Tense Consistency
* Use examples for better understanding

**Error Categories:**
1. **Awkward Word Choice and Phrasing**: Suggestions to enhance clarity and readability with alternative word choices or phrasing
2. **Cliche**: Hackneyed phrases or overly common imagery that lack originality or depth
3. **Poor Sentence Structure**: Feedback on sentence construction, recommending changes for better flow, clarity, or impact
4. **Unnecessary/Redundant Exposition**: Redundant parts of the text that can be removed or rephrased for conciseness
5. **Lack of Specificity and Detail**: Need to provide more concrete details or information to enrich and make the text engaging
6. **Purple Prose**: Unnecessary ornamental and overly verbose parts of the text that can be omitted
7. **Tense Consistency**: Identifying inconsistencies in tense to maintain uniformity throughout the paragraph.

**Instructions:**
- Select problematic spans and assign them to appropriate error categories.
- Spans must be verbatim from the paragraph.
- No overlap between spans.
- Each span should have exactly one category.

### A.2 Precision Metrics Explanation and Example

**Precision Metrics Explanation and Example**

**Illustrating Precision on a Simple Example**:
- System 1 and System 2 have produced different predictions for an annotated sentence: "On this dark and stormy night, her heart skipped a beat as she was afraid of what was to come."
- **Annotated spans**:
    - Span 1: characters [9,30]; category: CLICHÉ
    - Span 2: characters [57,94]; category: UNNECESSARY EXPOSITION
- **System 1**: predicted a single span that included both the annotated spans, but also additional characters
- **System 2**: predicted two separate spans that partially overlapped with the annotated spans

**Calculating General and Categorical Precision**:
- **General Precision**: overlap between predicted and annotated spans, divided by total predicted characters:
    - System 1: (30-9 + 94-57) / (94-9) = **0.68**
    - System 2: (30-19 + 94-57) / ((30-19) + (94-57)) = **1.0**
    - System 2 has higher general precision, as its predictions were entirely included in the annotated spans
- **Categorical Precision**: overlap is only considered valid if the overlapping spans have the same category:
    - System 1: (1*(30-9) + 0*(94-57)) / (94-9) = **0.25**
    - System 2: (1*(30-19) + 0*(94-57)) / ((30-19) + (94-57)) = **0.23**
    - System 1 had higher categorical precision by fully overlapping with the annotated CLICHÉ span, while System 2 only partially overlapped

**Limitations of Precision Scores**:
- Precision scores can be inflated by reducing predictions
- LLMs tend to select more spans than human annotators, leading to high recall but potentially lower precision

### A.3 Rewriting Prompts

**Rewriting Prompts:**

**Cliche Edits:**
- Replace overused phrases with more unique ones (e.g., "like creases in an old pocket map" instead of "etched into his being")
- Make sure edited text is coherent and grammatically correct with the following text

**Example 1:** Matthews had lived in the Valley all his life, and its rhythms and secrets were etched into his being like creases in an old pocket map. He knew the ins and outs of the place like no other.

**Poor Sentence Structure Edits:**
- Rewrite for clarity and proper sentence construction
- Ensure coherence and grammatical correctness with following text

**Example 4:** Z.’s laughter grew louder, his words slurring together like a sloppy melody. We exchanged a knowing glance; concern simmered beneath the surface as we looked on.

**Unnecessary/Redundant Exposition Edits:**
- Remove excessive explanations that don’t contribute to meaningfully to story or narrative
- Make sure edited text is coherent and grammatically correct with following text

**Example 13:** As I entered the quiet, garden-facing room on the second floor, stillness pervaded the space. The elderly couple sat motionless in their armchairs.

**Lack of Specificity and Detail:**
- Examples of paragraphs lacking specificity and detail (25 total)
- Rewrite suggested for each example to provide more concrete information
- Ensure coherence and grammatical correctness in edited text

**Purple Prose:**
- Excessive use of adjectives, adverbs, and metaphors can disrupt narrative flow
- Given examples of 25 paragraphs with purple prose to be rewritten
- Suggested edits for each example to simplify language or remove purple prose
- Ensure coherence and grammatical correctness in edited text.

