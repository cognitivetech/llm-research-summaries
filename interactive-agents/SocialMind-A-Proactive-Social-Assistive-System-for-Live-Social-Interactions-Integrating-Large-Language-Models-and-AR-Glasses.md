# SocialMind A Proactive Social Assistive System for Live Social Interactions Integrating Large Language Models and AR Glasses
source: https://arxiv.org/html/2412.04036v1
by Bufang Yang, Yunqi Guo, Lilin Xu, Zhenyu Yan, kai Chen, Guoliang Xing, Xiaofan Jiang

## Contents
- [Abstract.](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related work](#2-related-work)
  - [2.1. LLM-based Personal Assistants](#21-llm-based-personal-assistants)
  - [2.2. Social Assistive Systems](#22-social-assistive-systems)
  - [2.3. Proactive Conversational Systems](#23-proactive-conversational-systems)
  - [2.4. LLM Personalization and Acceleration](#24-llm-personalization-and-acceleration)
- [3. A Survey on Social Assistance Needs](#3-a-survey-on-social-assistance-needs)
  - [3.1. Design of Questionnaire](#31-design-of-questionnaire)
  - [3.2. Social Experience and Awkwardness](#32-social-experience-and-awkwardness)
  - [3.3. The Demand and Preference for a Social Assistant](#33-the-demand-and-preference-for-a-social-assistant)
  - [3.4. Privacy and User Comfort](#34-privacy-and-user-comfort)
  - [3.5. Findings Summary](#35-findings-summary)
- [4. System Design](#4-system-design)
  - [4.1. System Overview](#41-system-overview)
  - [4.2. Human-like Perception in Social Context](#42-human-like-perception-in-social-context)
  - [4.3. Implicit Persona Adaptation](#43-implicit-persona-adaptation)
  - [4.4. Multi-source Social Knowledge Reasoning](#44-multi-source-social-knowledge-reasoning)
  - [4.5. Multi-tier Collaborative Social Suggestion Generation](#45-multi-tier-collaborative-social-suggestion-generation)
- [5. Evaluation](#5-evaluation)
  - [5.1. Experimental Setup](#51-experimental-setup)
  - [5.2. Overall Performance](#52-overall-performance)
  - [5.3. Effectiveness of System Modules](#53-effectiveness-of-system-modules)
  - [5.4. Real-world Evaluation](#54-real-world-evaluation)
- [6. Discussion and Limitations](#6-discussion-and-limitations)
- [7. Conclusion](#7-conclusion)

## Abstract.

**Social Interactions and Virtual Assistants:**
- Social interactions are fundamental to human life
- Large language models (LLMs)-based virtual assistants have the potential to revolutionize human interactions and lifestyles
- Existing assistive systems mainly provide reactive services to individual users, rather than offering in-situ assistance during live social interactions with conversational partners

**Introduction of SocialMind:**
- SocialMind: first LLM-based proactive AR social assistive system
- Provides users with in-situ social assistance

**Features and Capabilities of SocialMind:**
- Employs human-like perception using multi-modal sensors to extract:
  - Verbal and nonverbal cues
  - Social factors
  - Implicit personas
- Incorporates these social cues into LLM reasoning for **social suggestion generation**
- Uses a **multi-tier collaborative generation strategy** and proactive update mechanism to display:
  - Social suggestions on Augmented Reality (AR) glasses
  - Ensure timely provision of suggestions without disrupting natural conversation flow

**Evaluation:**
- Achieves 38.3% higher engagement compared to baselines in three public datasets
- User study with 20 participants:
  - 95% willing to use SocialMind in live social interactions

## 1. Introduction

**Social Interactions and Quality of Life:**
- Social interactions significantly impact physical and mental health [^4]
- Over 15 million Americans experience social anxiety [^6]
- Assistance during live social interactions can enhance overall well-being for everyone

**Existing LLM-Based Virtual Assistants:**
- Focus on individual user service (writing, fitness, coding) [^25, 66, 22]
- Limited capability to provide assistance during live social interactions

**Proposed Social Assistive Systems:**
- Aim to support individuals with autism [^40], provide social etiquette consultation [^7], and resolve cultural conflicts [^36]
- Reactively address user queries or detect norm violations in conversations [^40, 36]

**Challenges in Developing a Proactive Social Assistive System:**
1. Provide instant and in-situ assistance during live interactions without disrupting the natural flow of conversation.
2. Understand nonverbal behaviors which are essential for social communication but challenging for LLMs to comprehend [^21].
3. Consider personal backgrounds and interests of both parties to enhance engagement [^79].
4. Integrate implicit personas into social suggestions [^21].
5. Develop a human-like perception mechanism that leverages multi-modal sensor data [^33].
6. Incorporate this information for reasoning and dynamically adjust strategies for providing instant social suggestions to the user [^79].

**Introduction of SocialMind:**
- First LLM-based proactive AR social assistive system that provides users with in-situ assistance during live social interactions
- Overview (Figure 1) shows the components and functionality of SocialMind.

**SocialMind Features:**
1. Multi-modal sensor data on AR glasses for human-like perception: verbal, nonverbal cues, and social factor information [^21].
2. Extract implicit personas through social interactions [^36].
3. Collaborative social suggestion generation strategy with a cache of social factor priors and intention infer-based reasoning approach [^79].
4. Proactive response mechanism displays suggestions on AR glasses without disrupting the natural flow of conversation.
5. Human-like perception mechanism to automatically perceive social cues [^33].
6. Multi-source social knowledge reasoning approach incorporates this information into LLM reasoning, dynamically adjusting strategies for social assistance.
7. Implicit persona adaptation approach generates customized social suggestions to enhance engagement during live interactions.
8. User survey with 60 participants revealed a 38.3% higher engagement compared to baselines and 95% willingness to use SocialMind in live social interactions [^21].

## 2. Related work

### 2.1. LLM-based Personal Assistants

**Voice Assistants:**
* Widely used in daily lives on various commercial mobile devices (Siri, Google Assistant) [^2][^29]
* LLM-based virtual assistants developed: fitness assistants [^66], writing assistants [^25], coding assistants [^22]
* OS-1 [^72]: virtual companion on smart glasses offering companionship and chatting with users
* UbiPhysio [^66]: fitness assistant providing natural language feedback for daily fitness and rehabilitation exercises
* Personal assistants for older adults [^26][^42] and individuals with impairments: EasyAsk [^26], Talk2Care [^76]
* LLM-based writing assistants: PEARL [^58], PRELUDE [^25]

**Single-User Interactions:**
* Systems focus solely on single-user human-to-computer interactions
* Consider only user's unilateral goals and inputs

**Social Assistance during Face-to-Face Interactions:**
* SocialMind provides users with social assistance during live, face-to-face interactions involving other parties.

### 2.2. Social Assistive Systems

**Pre-LLM Era: SocioGlass and Smart Glasses for Social Interaction Assistance**
* SocioGlass builds a biography database using smart glasses and facial recognition for social interaction assistance [^71]
* Study explores use of smart glasses to support social skills learning in individuals with autism spectrum disorder [^43]
* Limitations: display of social skills or biographies on-screen, lack context of real-time social conversation

**LLMs for Social Assistance: Advancements and Applications**
* Paprika employs LLMs to provide social advice to autistic adults in the workplace [^40]
  * Preferences interaction with LLMs
* Tianji: an LLM that comprehends social dynamics, offers guidance on social skill development [^7]
* Social-LLM integrates users’ profiles and interaction data for user detection [^41]
* SADAS checks user input for social norm violations to improve cross-cultural communication [^35]
* Kim et al develop a dialogue model to detect unsafe content and generate prosocial responses [^44]
* Systems provide post-assistance, addressing social norm violations after text input has been entered

**SocialMind: Proactively Providing Instant Social Suggestions in Live Face-to-Face Scenarios**
* SocialMind focuses on live face-to-face scenarios [^35]
* Perceives multi-modal nonverbal cues and conversation context to provide instant social suggestions
* Enables users to refer to suggestions before speaking

**Summary of Recent LLM-based Applications in Social and Communication** (● means included) | Approach | Base LLM | Social Assistance | Multi Party Interactions | Multi-modal Sensor Data | Persona- lization | Interactive Mode | System Settings | 
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Social-LLM** [^41] | Sentence-BERT | ○ | ○ | ○ | ○ | Reactive | PC |
| Paprika [^40] | GPT-4 | ● | ○ | ○ | ○ | Reactive | PC |
| Tianji [^7] | InternLM | ● | ○ | ○ | ○ | Reactive | PC |
| **SADAS** [^35] | ChatGPT | ● | ● | ○ | ○ | Reactive | HoloLens |
| OS-1 [^72] | GPT4, Gemini, Llama2 | ○ | ○ | ● | ● | Proactive | Glasses |
| PRELUDE [^25] | GPT-4 | ○ | ○ | ○ | ● | Reactive | PC |
| **SocialMind** | GPT-4o, Llama-3.1 | ● | ● | ● | ● | Proactive | Glasses |

### 2.3. Proactive Conversational Systems

**Proactive Conversational Systems:**
- **Receive and respond paradigm**: exemplified by writing assistants [^25] and coding assistants [^22]
  * Generate answer based on user's input
  * No further interaction
- **Proactive conversational systems**: can initiate and steer conversations through multi-turn interactions with users [^19]
  * Utilize historical context, perceived environmental information, expert knowledge, and sensor data for engagement

**Examples of Proactive Conversational Systems:**
- OS-1 [^72]
  * Virtual companion that utilizes personal daily logs and perceived environmental information to proactively engage users
- DrHouse [^75]
  * Multi-turn diagnostic system with expert medical knowledge and sensor data for assessments
- WorkFit [^9]
  * Proactive voice assistant that detects sedentary behavior in workers and generates interventions and health suggestions

**Limitations of Existing Proactive Conversational Systems:**
- Limited to individual user scenarios.
- Gap in research on proactive assistive systems for live social interactions involving conversational partners.

### 2.4. LLM Personalization and Acceleration

**LLM Caching:**
- **Caching solutions**: used to reduce repetitive computations in LLM reasoning systems
- Examples: GPT-cache, SCALM, key-value cache, CachedAttention, Prompt Cache
- **Semantic cache**: stored LLMs responses (GPT-cache and SCALM)
- **Key-value cache**: reusing attention states to reduce inference costs (numerous studies)
- **CachedAttention**: reuses KV cache of historical tokens in multi-turn conversations
- **Prompt Cache**: resues attention states of overlapped text segments among different prompts
- **SocialMind**: incorporates social factor priors into the cache to enhance accuracy

**Streaming and Real-time LLMs:**
- **Real-time AI assistants**: Gemini Live, supports users to interrupt conversations (supports daily tasks on mobile phones)
- **Real-time speech LLMs**: Mini-Omni, integrates hearing, speaking, and thinking into speech foundation models for real-time conversation; Speech ReaLLM, achieves real-time speech recognition by streaming speech tokens into LLMs for reasoning without waiting for the entire utterance or changing the LLM architecture
- **Limitations**: focus on general speech recognition, lack integration of multi-modal social knowledge, utility in live social interactions.

## 3. A Survey on Social Assistance Needs

We conducted a survey to understand users' experiences and needs for social assistants in live interactions, which informs our system's design.

### 3.1. Design of Questionnaire

**Questionnaire Sections and Topics:**
* **P1**: Social experiences:
  * Experiences of social awkwardness
  * Sources of social awkwardness
  * Attention to nonverbal behaviors during interactions

* **P2**: Virtual social assistance technologies:
  * Attitudes towards virtual assistance during live interaction
  * Preferred devices for assistance
  * Desired social situations and scenarios
  * Content of suggestion preferred
  * Format of assistive information
  * Tolerance for system latency
  * Information display methods

* **P3**: Privacy and comfort:
  * Attitudes towards interacting with users using virtual assistants
  * Concerns about personal data capture during interactions.

**Questionnaire Summary:**
- Comprises three sections, totaling 14 questions
- Designed to capture social experiences and attitudes towards virtual social assistance technologies
- Assesses participants' social awkwardness, preferences for devices, desired situations, content of suggestion, format of assistive information, tolerance for latency, information display methods, privacy concerns, and comfort with interaction using virtual assistants.

### 3.2. Social Experience and Awkwardness

**Social Awkwardness and Interactions:**
* 18.3% enjoy interacting with others; 81.7% report experiencing some level of awkwardness (Figure 2)
* Sources: workplace superiors/professors, formal events, peers, long-lost acquaintances, unfamiliar relatives
* Authority figures, formal settings, and unfamiliarity contribute significantly to social awkwardness
* Nonverbal behaviors play a crucial role in social interactions: facial expressions, tone of voice, personal distance, gestures (Figure 2)
* Majority considers nonverbal cues essential during interactions; challenges in interpreting indirect nonverbal behaviors
* Do you feel anxious or fearful during social interactions? (Figure 2)

**Survey Results:**
* Only 8.3% overlook nonverbal behaviors
* Facial expressions essential for 80.0%; tone of voice for 65.0%
* Personal distance important to 38.3%; gestures supplementary cues for 31.7%
* Significance of nonverbal behaviors in social interactions highlighted.

### 3.3. The Demand and Preference for a Social Assistant

**Social Awkwardness and Virtual Social Assistant:**
* Preference for virtual assistant aligns with social awkwardness (70.0% find it beneficial)
* Desire for assistance during uncertain situations:
  + Unsure about what to say (66.7%)
  + Interacting with specific individuals, especially authority figures (56.7%)
  + Initiating conversation or responding (48.3%)
* Content preferences for virtual assistant's suggestions:
  + Insights into conversational partners’ interests and backgrounds (70.0%)
  + Own interests and backgrounds (over 50.0%)
  + Updates on trending topics (40.0%)
  + Social cues about nonverbal behaviors (half)
* Preferred assistive device: glasses (56.7%)
* Information display preference: text in field of vision (93.3%)
* Assistive information format: summarized bullet points and example sentences (68.3%)
* Instant assistance preferred (90.0%)
* Demand for AR glasses for live social interactions.

### 3.4. Privacy and User Comfort

The adoption of virtual social assistance technologies is driven by openness, with 88.3% willing to engage. However, users become less comfortable when faced with privacy concerns, such as image capture. Despite this, 63.3% remain interested in continuing conversations.

### 3.5. Findings Summary

**Key Findings:**

**Social Awkwardness**:
- Common in daily life, especially with authority figures, formal settings, and unfamiliar situations
- Virtual social assistance has potential benefits

**Nonverbal Behaviors**:
- Gestures, facial expressions, personal distance are essential in interactions
- Effective virtual assistant should provide human-like perception for nonverbal cues

**Participant Interests**:
- Strong interest in a virtual social assistant that offers instant guidance to reduce social awkwardness
- Prefer assistance in specific scenarios, certain suggestion content, natural integration via glasses, and concise, instant suggestions

**Conclusion:**
- Clear demand for a proactive system based on AR glasses to provide effective social assistance during live interactions.

## 4. System Design

### 4.1. System Overview

**SocialMind: A Proactive Social Assistive System for Live Social Interactions**

**System Overview**:
- Leverages multi-modal sensor data (audio, video, head motion) for human-like perception in social contexts
- Automatically extracts nonverbal and verbal behaviors, parses social factor cues, and identifies implicit persons
- Integrates extracted data into Large Language Models (LLMs) for reasoning
- Employs multi-tier collaborative reasoning strategy with a social factor-aware cache and intention inferencing approach to generate in-situ social suggestions
- Displays suggestions on AR glasses through proactive response mechanism, assisting users in live social interactions without disrupting conversation flow

**Reasons for Using AR Glasses**:
1. Increasingly accepted for daily wear, as seen in applications like captioning and translation
2. Non-distracting solution, allowing users to maintain eye contact during social interactions
3. Preferred by most participants over other devices in a survey

### 4.2. Human-like Perception in Social Context

**SocialMind: A Proactive Social Assistive System**

**Background:**
- Focuses on live face-to-face social interactions
- Multi-modal sensor data for extracting social cues
- Challenges: high bandwidth usage, latency, privacy concerns

**Approach:**
1. **Nonverbal Cues Perception**:
   - Facial expressions and gestures indicative of emotional state or understanding
   - Processing local data using MediaPipe Holistic for pose estimation
   - Specialist models to generate nonverbal cues
   - Incorporating nonverbal cues into language models for social suggestions
2. **Efficient Primary User Identification:**
   - Challenges: voice fingerprinting, volume-based solutions
   - Lightweight approach using vibration signals on smart glasses
   - Calculate energy within 3 ∼ 10 Hz range as indicator
   - Grid search for optimal threshold and detection performance comparison with audio-based solutions
3. **Social Factor Cues Parsing:**
   - Social behaviors and speech vary based on social factors
   - Two modes of perception: reactive and proactive

**Nonverbal Cues Perception:**
* Facial expressions and gestures detected as nonverbal cues
* Processed locally using MediaPipe Holistic for pose estimation
* Specialist models used to generate additional information (§ [5.1.1](https://arxiv.org/html/2412.04036v1#S5.SS1.SSS1 "5.1.1. System Implementation ‣ 5.1. Experimental Setup ‣ 5. Evaluation ‣ SocialMind: A Proactive Social Assistive System for Live Social Interactions Integrating Large Language Models and AR Glasses"))

**Efficient Primary User Identification:**
* Challenges: voice fingerprinting and volume-based solutions (inadequate for live social interactions)
* Lightweight approach using vibration signals on smart glasses as indicator
* Calculate energy within 3 ∼ 10 Hz range as vibration signal indicator
* Set threshold for primary user detection at 1.1 on the server, sample rate is 466 Hz (§ [5.3.3](https://arxiv.org/html/2412.04036v1#S5.SS3.SSS3 "5.3.3. Impact of Hyper-parameter Settings. ‣ 5.3. Effectiveness of System Modules ‣ 5. Evaluation ‣ SocialMind: A Proactive Social Assistive System for Live Social Interactions Integrating Large Language Models and AR Glasses"))

**Social Factor Cues Parsing:**
* Social behaviors and speech vary based on social factors such as relation and formality
* Two modes of perception: reactive (automatic response) and proactive (anticipatory suggestions)
* Support for different social contexts in generating social suggestions.

### 4.3. Implicit Persona Adaptation

**SocialMind: Implicit Persona Adaptation for Personalized Social Suggestions**

**Unique Backgrounds and Interests**
- Each individual has unique backgrounds, life experiences, and personal interests
- These create "personas" that can enhance engagement in social interactions

**Challenges in Explicit Query Systems**
- Existing personal assistant systems rely on explicit queries for data retrieval
- Natural social conversations lack explicit queries
- Poses challenges for system personalization and implicit persona adaptation

**Implicit Persona Adaptation in SocialMind**
- Employs an additional LLM to extract implicit personas from historical conversations
- Maintains a persona database organized by individual identities
- Occurs during the post-interaction phase, where LLMs extract persona cues

**Persona Management Strategy**
- Live experiences and personal interests evolve over time
- SocialMind adapts to emerging personas through registration and merging/replacement of contradictory cues in the persona database

**Persona Retrieval**
- During live social interactions, SocialMind performs persona retrieval using face ID matching
- If conversational partners are found in the database, their personas are loaded as a knowledge source for customized social suggestions
- Otherwise, only the user’s persona is used.

### 4.4. Multi-source Social Knowledge Reasoning

**SocialMind: Knowledge Integration**

**4.4.1. Knowledge Source:**
- Contains nonverbal cues, context of live social conversations, social factors, implicit persona cues, and external tools (weather updates, social news)
- Multi-source and multi-modal information for social suggestion generation

**Nonverbal Cues:**
- Perception in Social Context (§ [4.2.1](https://arxiv.org/html/2412.04036v1#S4.SS2.SSS1))

**Context of Live Social Conversations:**
- Efficient Primary User Identification (§ [4.2.2](https://arxiv.org/html/2412.04036v1#S4.SS2.SSS2))

**Social Factors:**
- Social Factor Cues Parsing (§ [4.2.3](https://arxiv.org/html/2412.04036v1#S4.SS2.SSS3))

**Implicit Persona Cues:**
- Implicit Persona Extraction (§ [4.3.1](https://arxiv.org/html/2412.04036v1#S4.SS3.SSS1))

**External Tools:**
- Weather updates, trending social news

**4.4.2. Knowledge Integration:**
- Prompt consists of static and runtime portions

**Static Prompt:**
- Overall instructions, task instructions, prior knowledge (nonverbal cues usage guidelines, few-shot demonstrations, literature on utilizing nonverbal cues)
- Activates LLM's capability to generate social suggestions
- Enhances instruction following capabilities

**Runtime Prompt:**
- Dynamic portion that changes during live social interactions
- Integrates context of conversations, multi-modal nonverbal cues, implicit persona cues, and parsed social factors
- Enables SocialMind to adjust social suggestions in real time

**4.4.3. Concise CoT Reasoning:**
- Add instruction "Let's think step by step" into the prompt
- Employs concise Chain-of-Thought (CoT) reasoning strategy for social suggestion generation
- Sets constraints on generation length to 70 words or less
- Display format: summarized suggestions in bullet points followed by a sample sentence.

### 4.5. Multi-tier Collaborative Social Suggestion Generation

**Social Assistance during User Interactions:**
* Instant social suggestions required for smooth conversations without disrupting flow
* Multi-tier collaborative approach: social factor-aware cache and intention infer-based reasoning strategy (Figure [6](https://arxiv.org/html/2412.04036v1#S4.F6))

**Cache:**
* Widely used to avoid redundant computations but struggles with logical consistency in social interactions
* SocialMind leverages social factor priors for cache management: cache initialization, groups, selection, and runtime routing
* Cache initialization: simulated conversations between two LLM agents under various social factors
* Cache groups: conversations grouped based on social factors, indexed by these social factors
* Cache merging strategy: partially matched caches merged into a single group for caching
* Runtime routing and cache management: semantic similarity calculated using BERT, threshold set at 0.95, continuous recording of utterances, conversational partners' utterances, nonverbal cues, and social suggestions updated into the social factor-aware caches.

**Intention Infer-based Suggestion Generation:**
* Social factor-aware cache faces challenges in providing logically consistent social suggestions
* LLM reasoning employed for deep reasoning but causes significant system latency, disrupting natural flow of conversations
* Intention infer-based strategy inspired by human behaviors: initial understanding of intentions based on partially spoken words, early preparation for response
* Real-time speech recognition on AR glasses, offloading incomplete utterances to the server every 2 seconds to reduce bandwidth usage.

## 5. Evaluation

**Experimental Setup**

* Introduce the experiment
* Describe how SocialMind was evaluated 

**Evaluation & User Study**

* Report on the evaluation of SocialMind 
* Present a real-world user study

### 5.1. Experimental Setup

**SocialMind: A Proactive Social Assistive System for Live Social Interactions**

**5.1.1. System Implementation**:
- Off-the-shelf RayNEO X2 smart glasses as hardware platform:
  - Android 12 operating system, 6GB RAM, 128GB storage
  - Front-facing camera, dual-eye waveguide color displays, three microphones
  - Compatible with other AR glasses (e.g., INMO)
- On-glass app and Python-based server implementation:
  - 4,038 lines of Java and Kotlin code
  - Local video/audio processing using MediaPipe for efficient pose/landmark tracking
  - Azure Voice Recognition with local voice feature extraction
  - Vibration-based primary user identification due to privacy concerns
- Server handles social cue recognition and proactive suggestions:
  - Lightweight Scikit-learn models
  - Langchain for LLM coordination
  - HTTPS communication between glasses and server
  - Most code deployed locally on the glasses using Chaquopy, except for LLM inferences

**5.1.2. Experiments on Public Datasets**:
- Validation of SocialMind's effectiveness using public multi-turn dialogue datasets:
  - DailyDialog dataset (13,118 conversations)
  - Synthetic-Persona-Chat dataset (20,000 conversations, 5,000 personas)
  - SocialDial dataset (6.4K dialogues with social factor annotations)
- Use of two LLM agents for role-playing social interactions:
  - User agent interacts with partner agent while incorporating social suggestions
  - Randomly select 50 samples in each dataset for experiments

**Setup of Simulated Agents**:
- Enable multi-turn social conversations and nonverbal behaviors (facial expressions, gestures, physical proximity)
- Use personas as personal profiles and historical data for implicit persona extraction

**Two Role-play Paradigms**:
- Dialogue-based role-play: Agents initiate interactions using dialogues from three datasets
- Social factor-based role-play: Agents start interactions guided by social factors (norms, relations, formality, location)

### 5.2. Overall Performance

**SocialMind Performance Evaluation Results**

**Quantitative Results**:
- SocialMind outperforms baselines on:
  - Personalization: +38.7%
  - Engagement: +38.3%
  - Nonverbal cues utilization: +61.7%
- Reason for improvement: SocialMind incorporates multi-modal nonverbal cues, unlike conversation-only baselines
- Performance across different social scenarios: SocialMind achieves highest overall performance

**Qualitative Results**:
- Dialogues and social suggestions provided by SocialMind and baselines
- Observations:
  - Intention infer-based reasoning strategy enables logically consistent, instant social suggestions
  - Incorporation of conversational partner's nonverbal cues into social suggestions
  - Generation of customized social suggestions based on implicit persona cues for both parties

### 5.3. Effectiveness of System Modules

**Effectiveness of Social Factor Prior (SF-aware) in SocialMind:**
* Validated through experiments on simulated conversations using GPTcache as baseline
* Achieves 4.6% higher accuracy than GPTCache due to social factor-aware cache and utilization of social factor priors

**Impact of Sub-modules in SocialMind:**
* Implicit Personas Adaptation module and Nonverbal Cues Integration module significantly enhance Personalization, Engagement, and Nonverbal Cues Utilization in SocialMind

**Hyper-parameter Settings Impact on SocialMind:**

**Base LLMs:**
- GPT-4o achieves highest overall performance among various base models for generating social suggestions
- Llama-3.1-70B-Instruct performs comparably to GPT-4o, making it a promising solution due to open source nature and cost reduction

**Cache Size and Threshold:**
- Cache hit rate increases with cache size, reaching 36.3% under threshold of 0.95 when SocialMind is still unfamiliar with user's environment
- Employs relatively high threshold in cache for delivering relevant responses

**Primary User Detection:**
- Voice volume-based approach and vibration-based approach used for identification
- Energy threshold significantly affects Success Rate (SR) for both solutions when set to 0.1 for audio-based solution and 1.1 for vibration-based solution.

### 5.4. Real-world Evaluation

**SocialMind Performance Evaluation**
- **System Performance**:
  - Energy use capped at 3 fps, keeping power consumption under 2 watts
  - Support up to 70 minutes of use on a single charge
  - Manual activation allows for extended battery life
  - Data transfer rates below 100 KB/s over HTTPS
  - Pose/face tracking latency within 70 ms
  - Average latencies for cache: 50 ms, LLM processing: 2.8 s
- **User Study**:
  - Recruited 20 participants to test SocialMind in real-world conversations
  - Participants wore AR glasses equipped with SocialMind and engaged in social interactions
  - Evaluated system's effectiveness in helping users manage sudden social interactions

**Questionnaire Details**:
- 6-question user survey on experience with SocialMind:
  - **Q1**: Have you used an eyewear social assistant before?
  - **Q2**: Satisfaction with suggestions from the eyewear assistant
  - **Q3**: Acceptability of latency during live interactions
  - **Q4**, **Q5**: Willingness to use and interact with others using the system
  - **Q6**: Perception of the design's innovation and functionality

**Participant Feedback**:
- All participants were new to eyewear social assistants
- 85% found the system novel and practical
- 70% were satisfied with the suggestions, finding them helpful during interactions
- Over 90% found latency acceptable, not disrupting conversation flow
- Majority expressed willingness to use SocialMind in social situations
- Some participants found it useful for improving English, managing focus, and reducing cognitive load
- Feedback suggests potential market demand for the system.

## 6. Discussion and Limitations

**System Scalability:**
- SocialMind can be extended to multi-modal LLMs for nonverbal cue extraction
- Improves system's generalization and scalability
- Consider edge-cloud collaborative framework for resource limitations
- Scalable to multi-person scenarios with camera view identification of individual speakers
- Plan to incorporate user's facial expressions with hardware advances

**User's Nonverbal Cues:**
- Integrating wearer's facial expressions could enhance assistance quality
- Heavy AR goggles impact comfort significantly
- Current solution avoids heavy AR goggles for daily use
- Future plans: incorporating nonverbal cues from both user and conversational partners with Meta Orion hardware advances

**Next Steps:**
- Adapting solution for complex applications, including multi-person conversations and specific use cases (SAD, ASD)
- Collaboration with therapists and domain experts to incorporate tailored therapeutic insights
- Ensuring effective and responsive AR interventions for unique needs of user groups.

## 7. Conclusion

SocialMind is a proactive social assistive system for AR glasses that extracts social cues from multi-modal sensors. It uses a collaborative reasoning strategy to provide instant suggestions during live interactions, boosting engagement by 38.3% and earning user approval of 95%.

