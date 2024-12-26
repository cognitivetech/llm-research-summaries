# Traversing Emotional Landscapes and Linguistic Patterns in Bernard-Marie

by Arezou Zahiri Pourzarandi, Farshad Jafari
https://arxiv.org/html/2410.09576

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
  - [1.1 Research Motivation](#11-research-motivation)
  - [1.2 Objectives](#12-objectives)
  - [1.3 Scope of the Study](#13-scope-of-the-study)
- [2 Background](#2-background)
- [3 Methodology](#3-methodology)
  - [3.1 Text Extraction and Preprocessing](#31-text-extraction-and-preprocessing)
  - [3.2 Linguistic Analysis](#32-linguistic-analysis)
  - [3.3 Sentiment Analysis](#33-sentiment-analysis)
  - [3.4 Emotion Detection](#34-emotion-detection)
  - [3.5 Aggregation and Analysis](#35-aggregation-and-analysis)
- [4 Analysis and Findings](#4-analysis-and-findings)
  - [4.1 Dans la Solitude des Champs de Coton](#41-dans-la-solitude-des-champs-de-coton)
  - [4.2 La Nuit Juste Avant les Forêts](#42-la-nuit-juste-avant-les-forêts)
  - [4.3 Combat de Nègre et de Chiens](#43-combat-de-nègre-et-de-chiens)
  - [4.4 Comparative Analysis of Koltès’ Plays](#44-comparative-analysis-of-koltès-plays)
- [5 Discussion](#5-discussion)
  - [5.1 Literary Implications](#51-literary-implications)
  - [5.2 Methodological Reflections](#52-methodological-reflections)

## Abstract

**Study Overview:**
- Utilizes Natural Language Processing (NLP) to analyze Bernard-Marie Koltès' linguistic and emotional elements in contemporary French theatre plays.
- Incorporates advanced computational techniques to dissect his narrative style, revealing language-emotion interplay within his works.
- Findings reveal Koltès' crafting methods, enhancing our understanding of his thematic explorations, and contributing to digital humanities in literary analysis.

## 1 Introduction

**Bernard-Marie Koltès (1948–1989)**

**Renowned for:**
- Profound explorations of themes: isolation, power dynamics, communication, existential despair
- Rich psychological depth
- Complex characters
- Exploration of marginalized or liminal spaces
- Postcolonial identity and questions of national belonging

**Characteristics of his works:**
- Blend of poetic language and stark realism
- Transcend simple storytelling
- Vehicle for philosophical inquiry and social commentary

**Notable Plays:**
- Dans la Solitude des Champs de Coton
- La Nuit Juste Avant les Forêts
- Combat de Nègre et de Chiens

**Impact on Modern French Theatre:**
- Profound influence on modern French theatre
- Complex language and emotional depth make comprehensive literary analysis challenging using traditional methods.

### 1.1 Research Motivation

**Challenges in Interpreting Koltès' Works**
- Intricate linguistic structures: difficult to understand due to complex word choices and sentence construction
- Emotionally charged nature of dialogue: requires nuanced analytical approach beyond conventional literary critique
- Gap in understanding: how language functions structurally within Koltès’ plays, particularly in driving emotional engagement and dramatic tension
- NLP techniques used: sentiment analysis, emotion detection, linguistic feature extraction
- Goals: uncover patterns and structures that underpin psychological and emotional depth of Koltès' works
- Insights: explore emotional trajectories of characters and dramatic tension in narratives.

**Natural Language Processing (NLP) Approach**
- State-of-the-art techniques used for analysis
- Leverages tools like sentiment analysis, emotion detection, linguistic feature extraction
- Quantitative and systematic exploration of Koltès’ use of language
- Provides novel insights into emotional trajectories of characters and dramatic tension in narratives.

### 1.2 Objectives

**Research Objectives:**
- **Linguistic Patterns**:
  - Explore distinctive linguistic features of Koltès' texts: word frequency, vocabulary richness, Type-Token Ratio (TTR)
  - Reveal how Koltès' use of language contributes to thematic and atmospheric elements in his plays

- **Emotional Trajectories**:
  - Apply sentiment analysis and emotion detection models
  - Map evolving emotional landscapes within Koltès' narratives
  - Trace emotional peaks and valleys, contributing to overall dramatic tension
  - Understand characters' psychological depth and audience impact

- **Dramatic Tension**:
  - Quantify dramatic tension and pacing in Koltès' work through computational analysis
  - Elucidate mechanisms for building suspense and engaging audiences
  - Enhance understanding of Koltès' dramatic artistry.

### 1.3 Scope of the Study

**Bernard-Marie Koltès' Plays:**

**Dans la Solitude des Champs de Coton (In the Solitude of the Cotton Fields):**
- Tense, nocturnal encounter between a Dealer and a Client
- Set against desolate landscape reminiscent of cotton fields
- Explores themes: desire, commerce, inherent violence in transactions
- Ambiguous setting and poetic language universalize confrontation
- Metaphor for broader human experiences and interactions

**La Nuit Juste Avant les Forêts (The Night Just Before the Forests):**
- Monologue exploring loneliness, alienation, need for connection
- Fragmented speech reflects protagonist's disjointed thoughts and life
- Powerful narrative challenging audience to confront realities of exclusion and search for identity

**Combat de Nègre et de Chiens (Black Battles with Dogs):**
- Set in remote construction site in unnamed African country
- Examines complex dynamics of colonialism, racism, clash of cultures
- Intense dialogues and dramatic confrontations reveal underlying prejudices, desires, and fears of characters
- Critical look at lingering impacts of colonialism and nature of human conflict

**Approach to Literary Analysis:**
- Merges advanced NLP techniques with traditional dramaturgical study
- Focuses on linguistic patterns, emotional trajectories, dramatic tension
- Deepens understanding of Bernard-Marie Koltès’ work and other dramatic texts in a nuanced, computationally informed manner.

## 2 Background

**Emotion Dynamics and Sentiment Analysis in Literary Texts**

**Background**:
- Emotion dynamics and sentiment analysis have become increasingly prominent in literary studies
- Computational techniques used to explore emotional trajectories of characters and narratives
- Limitations of lexicon-based methods for capturing intricate emotional relationships between characters

**Emotion Analysis in Novels**:
- Distinguishing emotional arcs of characters from overall narrative
- Key insights into how characters' emotions evolve independently from the story [Vishnubhotla et al., 2024]

**Emotion Analysis in Dramatic Texts**:
- Employed word-emotion lexicons to map characters' emotional landscapes
- Revealed emotional contrasts that drive plot, especially in tragedies [Yavuz, 2020]
- Limitations: Static nature of lexicon-based approaches and failure to capture temporal shifts

**Character-to-Character Sentiment Analysis**:
- Highlighted dynamic aspect of character interactions
- Failure to fully capture nuanced emotional exchanges using traditional sentiment analysis tools [Nalisnick & Baird, 2013a]

**Sentiment Networks**:
- Tracked emotional exchanges between characters across an entire play or novel
- Revealed how emotional polarities develop and shift over time [Nalisnick & Baird, 2013b]
- Modeled relationships based on dialogues and interactions [Elson et al., 2010]
- Limited capacity to model more abstract and implicit emotional undercurrents in complex narratives

**Deep Learning Approaches**:
- Recent advancements offer new avenues for exploring emotion dynamics in literature
- Neural models capture nuanced emotional shifts within a narrative [Teodorescu & Mohammad, 2022]
- More sophisticated models required to understand the complexities of contemporary works like those of Bernard-Marie Koltès.

## 3 Methodology

**Preparing Koltès' Play Texts for Analysis**

Systematically extracted and cleaned texts for NLP processing. Methodology included: text extraction, cleaning, segmentation, application of various NLP tools and techniques for linguistic and emotional analysis. All codes are available at the [GitHub repository](https://github.com/frshdjfry/NLP-in-Dramatic-Literature).

### 3.1 Text Extraction and Preprocessing

**Text Extraction**: Used the French version of Tesseract OCR to convert printed pages of Koltès' plays into machine-readable text.

**Cleaning**: Removed OCR errors, irrelevant formatting, and non-textual elements for data purity.

**Text Segmentation**: Segmented plays into 150-word units, approximating one minute of stage time, for linguistic and emotional content analysis in performance-relevant chunks.

### 3.2 Linguistic Analysis

**Linguistic Analysis with spaCy (French)**
- Word Frequency/Word Clouds: Highlighting common words & thematic elements using matplotlib's WordCloud
- Vocabulary Richness: Assessing linguistic complexity and stylistic variance through lexical diversity computation

### 3.3 Sentiment Analysis

- **Used sentiment analysis pipeline (Hugging Face's transformers library with tblard/tf-allocine model fine-tuned for French texts [Blard, 2020](https:**//arxiv.org/html/2410.09609v1#bib.bib1))
- Assessed sentiment of each text segment to analyze plays' emotional arcs and dramatic tension

### 3.4 Emotion Detection

**Emotion Analysis**: Utilized bhadresh-savani/bert-base-uncased-emotion model (Hugging Face's pipeline) for text classification [Tunstall et al., 2022]. This model provides multiple emotion scores, generating a detailed emotional profile for each segment.

### 3.5 Aggregation and Analysis

**Methodology for Sentiment and Emotion Analysis**

Averaged scores within each 150-word segment to get a representative emotional or sentiment value for a "minute" of the play, enabling plotting of emotional intensity and sentiment valence progression throughout plays. This approach offers novel insights into Koltès' dramaturgy and thematic layering by visualizing emotions and sentiments over time.

This methodology blends traditional literary analysis with modern NLP techniques, providing a comprehensive exploration of Koltès' works that merges qualitative literary insights with quantitative linguistic and emotional patterns.

## 4 Analysis and Findings

### 4.1 Dans la Solitude des Champs de Coton

**Insights from "Dans la Solitude des Champs de Coton" Analysis:**
- **Type-Token Ratio (TTR)**: 0.4919 indicates a diverse vocabulary
- Complex interplay of themes: desire, identity, human-animal dichotomy

**Vocabulary Analysis:**
- **Desire**: frequent use of "désir" without specifying object
- Exploration of unarticulated desires
- Human and Animal: juxtaposition of "homme" (human) and "animal"
- Blurred lines between human and animalistic instincts
- Time and Place: recurring references to "heure" (hour), "temps" (time), and "point" (place)
  - Abstract setting with no specific temporal or spatial anchors
- Coldness: pervasive motif of "froid" (cold)
  - Chilling atmosphere
  - Symbolizes emotional coldness and existential isolation

**Sentiment Analysis:**
- Dramatic tension intensifies in final third of play
- Dialogue becomes increasingly terse, reflecting a climax in conflict
- Emotional analysis: increase in "sadness" and "fear" towards the end
  - Crescendo in dramatic tension and existential dread.

### 4.2 La Nuit Juste Avant les Forêts

**Vocabulary Richness Analysis:**
- **Type-Token Ratio**: 0.3624 (indicates limited vocabulary range)
- **Word Cloud**:
  - Most frequent terms: "nuit" (night), "mec" (guy), "con" (idiot), "chambre" (room), "coup" (shot), "camarade" (comrade), "rue" (street), "gueule" (mouth), "monde" (world), "pute" (whore)
  - Visual representation of narrative in a nocturnal, urban setting

**Narrative Focus:**
- Contrast between "chambre" and "rue": longing for shelter vs. reality of wandering streets
- Narrative oscillates between past and present: themes of rejection and camaraderie

**Emotional Landscape:**
- Sentiment analysis: significant moments of sadness and anger intensifying towards the end
- Protagonist's increasing desperation and anger highlighted in emotional depth and complexity
- Emotion analysis over time (Figure 7) and percentage distribution of emotions (Figure 8, 9): nuanced emotional landscape underlying narrative.

### 4.3 Combat de Nègre et de Chiens

**Linguistic Patterns: Type-Token Ratio (TTR) and Thematic Exploration**
* TTR of 0.3561 indicates focused thematic exploration in "Combat de Nègre et de Chiens"
* Recurrent use of "femme" (woman): themes of objectification, mystification in male-dominated environment
* Vocabulary analysis (Table 3) and word cloud (Figure 3) provide insight into linguistic landscape

**Setting and Conflict**
* Setting established through term "chantier" (construction site), signifying backdrop for drama and cultural clashes
* Frequent mentions of "corps" (body) and "tête" (head): physical conflicts, psychological battles

**Character Dynamics**
* Gender dynamics and power structures: focus on "monsieur" (sir) and "femme"
* Female presence pivotal role in escalating tension among male characters

**Emotional and Sentiment Analysis**
* Fluctuating emotional states throughout play illustrated by sentiment analysis (Figure 10)
* Emotion analysis (Figure 11) and percentage distribution of emotions (Figure 12) reveal complexities of character interactions, psychological landscapes.

### 4.4 Comparative Analysis of Koltès’ Plays

**Comparative Analysis of Koltès' Plays**

**Vocabulary Richness and Linguistic Diversity:**
- Dans la Solitude: Type-Token Ratio (TTR) of **0.4919**, indicating high vocabulary diversity
- La Nuit Juste Avant les Forêts and Combat de Nègre et de Chiens have lower TTRs of **0.3624** and **0.3561** respectively
- Lower TTR in the latter plays could be due to focused narrative scope or monologue format

**Recurring Themes:**
- Word frequency analysis reveals:
  - Dans la Solitude: "désir" (desire) appears **39 times**, highlighting theme of unspoken desires and transactions
  - La Nuit Juste Avant les Forêts: focus on "nuit" (night), appearing **40 times**, indicative of personal despair and societal alienation
  - Combat: emphasis on "femme" (woman), appearing **74 times**, reflecting themes of confrontation and cultural clashes

**Emotional Trajectories and Dramatic Tension:**
- Dans la Solitude: notable increase in dramatic tension towards the end, with more intense exchanges between characters
- La Nuit: emotional intensity is more evenly distributed, punctuated by peaks of "sadness" and "anger" towards climax
- Combat: consistent thematic focus on conflict and power dynamics, reflected in sustained presence of "anger" and "fear"

**Emotional Composition:**
- Dans la Solitude: balanced mix of **"anger" (42.8%)** and **"fear" (22.9%)**, suggesting a narrative fraught with conflict and uncertainty
- La Nuit: moments of **"sadness"** peaking towards conclusion, indicating crescendo of personal despair
- Combat: dominated by **"anger" (54.1%)**, reflecting themes of confrontation and cultural clashes

**Synthesis:**
- Koltès' ability to navigate a broad spectrum of human emotions and experiences through diverse linguistic choices
- Dans la Solitude: introspective and philosophical narrative style
- La Nuit: intimate look at personal anguish and societal alienation
- Combat: starkly realistic and confrontational approach to power, violence, and cultural tensions.

## 5 Discussion

### 5.1 Literary Implications

**Understanding Bernard-Marie Koltès' Plays through NLP Analysis**

**Insights from NLP Analysis**:
- Reveals thematic preoccupations and narrative techniques of Koltès' plays
- Highlights adaptive linguistic style tailored to unique emotional demands of each narrative
- Illuminates weaving of complex psychological landscapes using language for atmosphere and tension

**Vocabulary Richness Variation**:
- Across the plays, indicating Koltès' mastery in exploring human condition
- Adaptive linguistic approach, not limited to dialogue

**Word Frequency and Emotional Content Analysis**:
- Illustrates ability to create intricate emotional experiences through language

**Significance of Koltès' Work**:
- Challenges audiences to confront multifaceted realities of existence
- Pivotal explorations of contemporary life and its discontents

### 5.2 Methodological Reflections

**Intersection of Computational Linguistics and Literary Scholarship**
- **Analysis of literary texts using NLP**: Novel approach to processing large volumes of text systematically, uncovering patterns and trends
- **Strengths**: Ability to provide quantifiable metrics for sentiment analysis and emotion detection, gauging dramatic tension and emotional depth

**Limitations:**
- Subtleties of literary language can present challenges: Irony, metaphor, rhetorical devices may lead to misinterpretations
- Cultural and historical nuances require nuanced understanding that purely computational methods might miss

**Approach for Mitigating Issues:**
- **Manual verification**: Ensuring computational findings are contextualized within a broader literary framework
- **Interdisciplinary methods**: Combining computational techniques with humanistic interpretation to enhance literary studies.

**Potential of Interdisciplinary Approach:**
- Recognizing the inherent complexities of literary texts.

