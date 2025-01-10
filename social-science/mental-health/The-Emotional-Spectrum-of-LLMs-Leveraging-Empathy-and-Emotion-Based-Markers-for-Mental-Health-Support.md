# The Emotional Spectrum of LLMs Leveraging Empathy and Emotion-Based Markers for Mental Health Support

source: https://arxiv.org/html/2412.20068v1
by Alessandro De Grandi, Federico Ravenda, Andrea Raballo, Fabio Crestani

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Works](#2-related-works)
- [3 Methodology](#3-methodology)
  - [3.1 A Psychological Rationale for our Approach](#31-a-psychological-rationale-for-our-approach)
  - [3.2 Datasets](#32-datasets)
  - [3.3 Tailoring an LLM to React Empathetically](#33-tailoring-an-llm-to-react-empathetically)
- [4 Emotion Recognition and Empathetic Response](#4-emotion-recognition-and-empathetic-response)
- [5 Mental State as Mixtures of Emotions](#5-mental-state-as-mixtures-of-emotions)
- [6 Reddit’s Emotion embeddings applied to the Detection of Suicide Risk](#6-reddits-emotion-embeddings-applied-to-the-detection-of-suicide-risk)
  - [6.1 Results for Mental Health Classification](#61-results-for-mental-health-classification)
- [7 Conclusions](#7-conclusions)
- [8 Limitations](#8-limitations)
- [9 Ethical Considerations](#9-ethical-considerations)
- [Appendix A Finetuning Details](#appendix-a-finetuning-details)
- [Appendix B Error Analysis](#appendix-b-error-analysis)
- [Appendix C Beyond the 32 Emotion Classes](#appendix-c-beyond-the-32-emotion-classes)
- [Appendix D Qualitative results and analysis](#appendix-d-qualitative-results-and-analysis)
  - [D.1 Dealing with coexisting emotions](#d1-dealing-with-coexisting-emotions)
  - [D.2 Dealing with different intensities of the same emotion](#d2-dealing-with-different-intensities-of-the-same-emotion)
  - [D.3 Dealing with uncertainty](#d3-dealing-with-uncertainty)
- [Appendix E Reddit’s Subreddits](#appendix-e-reddits-subreddits)

## Abstract

**Psychological Conversational AI for Mental Health Support**

**Increasing Demand**:
- Highlighted need for innovative solutions in mental health services
- Particularly in realm of psychological conversational AI due to scarce sensitive data availability

**Exploring a Novel Approach**:
- Developing a system tailored for mental health support
- Based on explainable emotional profiles and empathetic conversational models
- Offering a tool to augment traditional care where expertise is unavailable

**Two Main Parts**:
1. **RACLETTE**: 
   - Conversational system demonstrating superior emotional accuracy
   - Understanding users' emotional states and generating empathetic responses
   - Progressively building an emotional profile of the user through interactions
2. **Emotional Profiles for Mental Health Assessment**:
   - Emotional profiles as interpretable markers for mental health assessment
   - Comparing profiles with characteristic patterns associated with different mental disorders
   - Providing a novel approach to preliminary screening and support

**The Emotional Spectrum of LLMs**:
- Leveraging empathy and emotion-based markers for mental health support

**Contributors**:
- **Alessandro De Grandi**: First author, Università della Svizzera italiana
- **Federico Ravenda**: First author, Università della Svizzera italiana
- **Andrea Raballo**: Contributor, Università della Svizzera italiana
- **Fabio Crestani**: Contributor, Università della Svizzera italiana

## 1 Introduction

**Empathetic Chatbots: A New Era of Conversational AI**

**Advancements in Empathetic Chatbots**:
- Designed to understand and respond to emotional states of users
- Leverage Natural Language Processing (NLP) approaches to analyze text for emotional content
- Engage in interactions that feel more human-like

**Benefits of Empathetic Chatbots**:
- Recognize and respond to a wide range of emotions
- Tailor responses to provide comfort, advice, or support
- Enhance user experience in applications like customer service, mental health support, personal assistants, and social companions

**Advanced Large Language Models (LLMs)**:
- Interaction experience with conversational agents has seen remarkable improvements
- Exhibit enhanced understanding of natural language, greater contextual awareness, and ability to generate more coherent responses

**RACLETTE: A Conversational System for Emotional Detection and Response**:
- Developed to detect, understand, and respond to emotional cues similar to human empathy
- Uses an unconventional 3-turn structure where the model is trained to predict the user's emotion as a next-token prediction
- Responds empathically based on the predicted emotion, updating the user's emotional profile in real-time

**Emotional Profiles and Mental Health Disorders**:
- Emotional states are interconnected components forming distinct patterns linked to various mental health conditions
- Pre-calculated emotional profiles for specific disorders can be compared with users' emotional profiles to differentiate between conditions, aiding in early detection and diagnosis

## 2 Related Works

**Research on Developing Empathetic Chatbots:**
* Significant efforts to create sophisticated conversational models that understand human emotions [^17]
* Importance of emotional understanding for enhancing human-computer interaction [^35]
* Recent research focuses on personalized conversational systems with coherence and user engagement [^34]
* Positive outlook on mental health chatbots, but need for enhanced linguistic capabilities and personalized interactions [^1]

**Application of NLP in Psychology:**
* Growing interest in using NLP to extract valuable insights from communication in mental health [^27]
* Language coherence as predictor of psychotic symptoms [^24]
* Clearer language production deficits observed during first episode of psychosis [^19]
* Core symptom, language disorganization, assessed using topic models [^5]
* Emotional markers as potential early indicators or diagnostic aids for mental disorders.

**Inspiration from Previous Research:**
* CAiRE's empathetic neural chatbot model [^31]
* Using grayscale labels for emotion recognition suggested by "The Emotion is Not One-hot Encoding" [^28].

## 3 Methodology

This work proposes a novel methodology for synthesizing empathetic responses using a broader understanding of affective language, allowing models to detect emotions and create explainable emotional profiles for mental health assessments without relying on sensitive conversational data.

### 3.1 A Psychological Rationale for our Approach

**Empathy: Two Main Components**
- **Cognitive Empathy**: intellectual ability to understand another person's emotions, thoughts, and motives
	+ Essential for effective communication and social interaction
	+ Involves understanding someone else’s mental state and why they might be feeling a certain way
- **Affective Empathy**: ability to physically feel another person’s emotions
	+ Leads to emotional responses such as compassion or concern
	+ Raises ethical and philosophical questions about consciousness and emotion in artificial systems

**Research Focus: Cognitive Empathy for Emotion Classification**
- Aim is to classify emotional state of a patient
- Enables system to respond appropriately

**Emotion Embeddings**
- High-dimensional vectors representing an emotional state
- Synthesize emotional information within a conversation
- Enable meaningful algebraic operations

**Complex Emotions Representation**
- Accumulated through summation of emotion embeddings from interactions

**Emotional Profiles and Mental Health Support**
- Emotion profiles used for mental disorder assessment is not new in psychometrics
- Aligns with established psychological assessment methods, like Beck Depression Inventory-II (BDI-II)
- Offers comprehensive view of individual’s mental state by capturing interplay of various emotions
- Reveals emotions as indicators of deeper, complex mental states
- Multidimensional approach acknowledges the complexity of human psychology for accurate and personalized mental health support.

### 3.2 Datasets

**RACLETTE Workflow:**

**Datasets Used:**
- Empathetic Dialogues Dataset [^38]:
  * Trains RACLETTE model for emotion identification and empathic response
  * Large-scale multi-turn conversation dataset from Amazon Mechanical Turk
  - Contains 24,850 one-to-one open-domain conversations
  - Wide range of emotions considered (not limited to a few fundamental ones)

**Reddit Mental Health Dataset [^33]:**
  * Constructs discrete distributions for mental disorders
  * Extracts emotion embeddings from specific Reddit subreddits
  * Table 3 shows the list of considered subreddits

**DailyDialog Dataset [^30]:**
  * Control group for emotional profiles assessment
  * Collection of dialogues: training set (11,118), validation (1,000), test sets (1,000)

**RACLETTE Workflow Pipeline:**
1. Training the empathetic model using Empathetic Dialogues Dataset
2. Extracting emotion embeddings from Reddit Mental Health Dataset
3. Comparative analysis using all datasets: control group (DailyDialog), mental disorders (Reddit Mental Health), and conversations (Empathetic Dialogues)

**Figure 3:**
- Overview of the main steps in RACLETTE pipeline
- Illustrates integration of different datasets into system's architecture.

### 3.3 Tailoring an LLM to React Empathetically

**Study Approach:**
- Chose Mistral 7B model (7-billion-parameters LLM) for fine-tuning on Empathetic Dialogues Dataset
- Aligns with recent findings from [^40] to reduce hallucinations in LLMs
- Leverages causal attention mask of transformer decoder model for empathic response prediction and dialogue emotion detection tasks
- Places prompts before emotion labels during training and inference, enforcing autoregressive property
- Model learns to predict next tokens by attending only to previous positions in sequence
- Objective: max log P(y\_<emotion>|y\_1,y\_2,…,y\_N), where N is length of sequence
- During prediction, model iteratively produces tokens most likely given previous tokens
- Uses Top-K Sampling to limit sampling pool and generate diverse set of emotions
- Aggregates emotional profiles across entire conversation for speaker analysis

**Study Methodology:**
- Fine-tuning 7B Mistral model on Empathetic Dialogues Dataset
- Use case: empathic responses prediction and dialogue emotion detection tasks
- Model attends to previous tokens in context during prediction: whole history of conversation, current prompt, and emotion
- Objective: generate appropriate reply based on training dataset.

**Emotional Spectrum of LLMs:**
- Leveraging Empathy and Emotion-Based Markers for Mental Health Support
- Unconventional 3-turns structure: Prompt, Emotion, Response (separated by <|prompter|>,<|emotion|>,<|assistant|>, and <endoftext>)
- Model attends to previous tokens in context when predicting empathetic responses.

## 4 Emotion Recognition and Empathetic Response

**Emotion Recognition and Empathetic Response: The Emotional Spectrum of LLMs**

**Table 1**: RACLETTE's Performance on Empathic Dialogues Dataset
- **Emotion Detection Accuracy**:
  - At conversation level: 56% → 59% (3% increase)
  - At prompt level: N/A
- **BERTSCORE Evaluation**:
  - Indicates high semantic similarity between model's responses and human replies (0.87)

**Table 2**: Comparison with Other Approaches on Empathic Dialogues Dataset
- **RACLETTE vs. CAiRE and Baselines**:
  - RACLETTE outperforms other approaches in emotion recognition accuracy
- **Benchmarks for Emotion Classification Accuracy**:
  1. Cause-aware empathetic generation using Chain-of-Thought fine-tuning on Large Language Models (not specified)
  2. Knowledge-enhanced empathetic dialogue generation with external knowledge and emotional signals (90.7%) [^29]
  3. Incorporating emotion cause recognition into empathetic response generation using an emotion reasoner and gated attention mechanism (86.4%) [^18]

**Table 1**:
- **Precision**, **Recall**, **F1-Score**, **Support**, and **Accuracy** for individual prompts (5,242) and conversations (2,472)

**Table 2**:
- Comparison of emotional accuracy between RACLETTE and other benchmarks

## 5 Mental State as Mixtures of Emotions

**Approach to Create Explainable Mental State Embeddings Based on Emotions**
* Novel approach expands conversational model's role from empathetic response generation to emotion analysis and diagnostic tool
* Leverages fine-tuned model as an emotion classifier
* Extracts emotional embeddings from specialized corpora, such as social media interactions in mental health forums
* Goal: compare distinctive emotional profiles obtained from users interacting with the model in conversation

**Data Collection**
* Reddit: social news website and forum where content is socially curated and promoted through voting
* Subreddits: focused on specific topics, interests, or themes, creating unique communities within the broader platform
* Considered subreddits reported in Table [3](https://arxiv.org/html/2412.20068v1#S6.T3) and visually represented in Figures [5](https://arxiv.org/html/2412.20068v1#A5.F5) and [6](https://arxiv.org/html/2412.20068v1#A5.F6) in Appendix
* Data obtained from Reddit may not accurately represent the broader population with mental illnesses, as it only captures those who choose to discuss their experiences online

**Methodology**
* Embeddings for each mental disorder generated by processing posts from respective subreddits
* Empathetic conversational model obtained in previous experiment used, but responds without reply; phrases segmented and emotions predicted instead
* Emotions predicted aggregated across all posts, normalized to obtain characteristic emotional distribution profiles for each mental disorder

**Results**
* Obtained emotion embeddings show significant differences across a spectrum of Reddit communities
* Similarities expected between related disorders: depression and suicide, addiction and alcoholism
* Mental disorder representations in two-dimensional reduced space (Figure [4](https://arxiv.org/html/2412.20068v1#S5.F4)): alcoholism, addiction, eating disorder close; depression, loneliness, schizophrenia, PTSD near each other
* Bipolar representation close to sum of depression and schizophrenia's embeddings

**Comparison with Normal Distribution**
* Establish normal distribution for comparison: Daily Dialogue Dataset (high-quality multi-turn open-domain English dialogue dataset) used as control group
* Average of 8 speaker turns per dialogue, around 15 tokens per turn; whole training set extracted embedding for this dataset
* Contrast in emotional profiles between reddit depression community and individuals engaging in daily dialogues emphasized (Figure [4](https://arxiv.org/html/2412.20068v1#S5.F4))

## 6 Reddit’s Emotion embeddings applied to the Detection of Suicide Risk

**Emotion-Based Suicide Risk Detection: Emotional Profiles Embeddings Comparison**
* Experiment evaluates use of emotion for suicide risk diagnosis using Reddit data
* Focuses on Emotional Profiles Embeddings (EPEs) comparison with suicide embedding
* Dataset: CasualConversation vs. SuicideWatch subreddits, 5% test set (≈10,585 samples)
* Removal of anomalous or non-informative posts
* Sentences divided into 10 emotions each, final embedding aggregates all sentences
* Metrics for comparing embeddings: Kullback–Leibler (KL), Jensen–Shannon (JS), Cosine Similarity (CS)

**Suicide Risk Embeddings Comparison: Emotion Embeddings vs. Suicide Embedding**
| **Emotion Embeddings** | **Comparison Metrics** | **Findings** |
|---|---|---|
| Suicide, Depression, BPD, Bipolar Disorder, Addiction, PTSD, Schizophrenia (Positive labels) | KL Divergence, JS Divergence, Cosine Similarity (CS) | Emotional proximity between suicide and depression; high similarity among related mental health conditions like BPD, Bipolar Disorder, Addiction, PTSD, and Schizophrenia |
| Normal, Uniform Distributions (Negative labels) |  | Control group for comparison |

**Experiment Focus**: Unsupervised fashion, map positive label EPEs to predicted label based on the closest match to suicide embedding.

### 6.1 Results for Mental Health Classification

**Table 4: Performance Metrics for Mental Health Classification**
- **Precision**: Measure of the model's ability to correctly identify positive instances
- **Recall**: Measure of the model's ability to identify all relevant cases, prioritizing severe consequences
- **F1 Score**: Harmonic mean of precision and recall, balancing both metrics
- **Accuracy**: Percentage of correctly classified instances

**Combined Method for Suicide Risk Detection:**
- If any method detects a risk of suicide, the label is assigned as positive to maximize recall
- Results in high recall at the cost of other metrics (precision, false positives)

**Performance Comparison:**
- **RACLETTE’s Combined Method**: Highest recall (0.95), superior ability to identify relevant cases
- **RoBERTa**: Highest precision (0.72), lower recall (0.84)
- **KL Divergence, JS Divergence, Cosine Similarity**: High recall (0.93 for JS Divergence and Cosine Similarity, 0.90 for KL Divergence), but lower precision

**Balanced Performance:**
- RACLETTE's **KL Divergence variant**: Strong scores across all metrics (precision: 0.71, recall: 0.90, F1: 0.79, accuracy: 0.77)

**Explainable Representations:**
- RACLETTE's methods generate explainable representations and emotion embeddings for valuable insights into emotional profiles.

**Benchmark Models:**
- Compared with state-of-the-art unsupervised approaches based on RoBERTa and BERT’s embedding representations, using K-Means clustering approach.

## 7 Conclusions

**RACLETTE System: Addressing Challenges in Mental Health Support**

**Challenge 1:** **Empathetic AI-driven conversation**
- Fine-tuned Language Model (LLM) creates effective conversational agents
- Recognizes users' emotional states
- Generates high-quality empathetic responses
- Avoids use of sensitive clinical data

**Challenge 2:** **Reliable assessment tools**
- Achieves state-of-the-art performance in emotion recognition
- Introduces novel methodology for creating emotional profiles
- Emotional profiles generated by aggregating emotion distributions from user interactions
- Interpretable markers for mental health condition analysis
- Effective in maintaining empathetic conversations
- Potential as a preliminary screening tool through the analysis of emotional embeddings.

## 8 Limitations

**Quality and Reliability of Emotional Data**
- Critical point: Ensure diverse and accurately labeled emotional data for model training
- Complex data collection process: Manual annotation by experts to maintain high standards
- Individuals with mental disorders use social media for sharing experiences, seeking info, and finding support [^36]
- Limitations of self-reported information: Personal biases, misunderstandings, or intentional misreporting
- Online self-expression varies greatly between individuals due to cultural differences, personal communication styles, context
- Confounding factors: Comorbidities can result in complex emotional and psychological profiles for the model to parse accurately
- Differences between online interactions and in-person communications add complexity to interpreting emotions

**Privacy, Confidentiality, Ethical Implications**
- Methodology addresses crucial privacy and confidentiality issues important in mental health domain
- Does not fully address ethical implications of using AI as a clinical tool: potential for misuse, need for safeguards against harmful or biased behaviors in conversational model

**Continuous Improvements and Validation**
- Continuous improvements and validation against clinical standards necessary to ensure effective integration into traditional care pathways
- Enhance therapeutic process rather than disrupting it.

## 9 Ethical Considerations

**AI Ethical Considerations for Mental Health Support and Assessment**

**Ethical Risks:**
- **Misuse as a clinical tool**: potential harmful or biased behaviors leading to adverse outcomes
- Ensuring responsible deployment: implementing ethical safeguards

**Addressing Ethical Risks:**
1. **Clear guidelines on use**: appropriate application and limitations
2. **Sensitive data protocols**: managing privacy and security concerns
3. **Transparency in operations**: being open about AI's capabilities and limitations
4. **Involvement of ethicists and clinicians**: creating balanced approach to development
5. **Communicate supplementary nature**: emphasizing the role of professional mental health care providers
6. **Integration into traditional care pathways**: maintaining central role of human practitioners
7. **Preventing over-reliance on automated tools**: encouraging users to seek help from qualified professionals.

## Appendix A Finetuning Details

The study employed SFTTrainer and QLoRa from HuggingFace libraries[^46]. Model parameters were quantized to 4-bit NormalFloat(nf-4) and computations to 16-bit BrainFloat (bFloat16). LoRa hyperparameters used: lora_alpha=16, lora_dropout=0.1, lora_r=64. Training hyperparameters: batch_size=1, gradient_accumulation_steps=16, warmup_ratio=0.3, cosine learning rate scheduler with l_r=2e-5, training for 3 epochs with AdamW optimizer.

## Appendix B Error Analysis

**Emotional Accuracy of Model (Empathetic Dialogues Dataset)**
* **Individual Prompt Level Analysis**: Table [5](https://arxiv.org/html/2412.20068v1#A2.T5 "Table 5 ‣ Appendix B Error Analysis ‣ The Emotional Spectrum of LLMs: Leveraging Empathy and Emotion-Based Markers for Mental Health Support")
	+ Overall good emotional accuracy
	+ Some emotions exhibit suboptimal performance
* **Conversation Level Analysis**: Table [6](https://arxiv.org/html/2412.20068v1#A2.T6 "Table 6 ‣ Appendix B Error Analysis ‣ The Emotional Spectrum of LLMs: Leveraging Empathy and Emotion-Based Markers for Mental Health Support")
	+ Overall assessment of model's performance
	+ Certain emotions challenging to differentiate in human evaluation
* **Specific Emotions**: 'angry' vs. 'furious'
	+ Technically distinct but share similar underlying sentiments
	+ Often interchangeable in practice.

## Appendix C Beyond the 32 Emotion Classes

**Emotion Classification: RACLETTE's Ability to Understand Emotions Beyond Training Data**

**Understanding the Broader Range of Emotions:**
- Empathic conversational agent aims for greatest possible range of emotion recognition (32 classes as minimum)
- Table [7](https://arxiv.org/html/2412.20068v1#A3.T7 "Table 7 ‣ Appendix C Beyond the 32 Emotion Classes ‣ The Emotional Spectrum of LLMs: Leveraging Empathy and Emotion-Based Markers for Mental Health Support") shows RACLETTE's ability to predict emotions not in fine-tuning dataset
- New emotions predicted when prompts contain explicit references (generalization)
- Model understands context and learned representations of emotions are similar

**Generative Pre-trained Transformer Decoder Model:**
- Unconventional use as a classifier
- Allows for flexibility compared to conventional classification approaches
- Enforces similarity in learned representations during fine-tuning process

**Out-of-Training-Labels Predictions:**
- Table [7](https://arxiv.org/html/2412.20068v1#A3.T7 "Table 7 ‣ Appendix C Beyond the 32 Emotion Classes ‣ The Emotional Spectrum of LLMs: Leveraging Empathy and Emotion-Based Markers for Mental Health Support") shows examples of new emotions predicted by RACLETTE as outliers but showcasing potential for generative models in classification tasks.

## Appendix D Qualitative results and analysis

The following examples compare the model's predictions against the targets for qualitative analysis using the Empathetic Dialogues dataset's test set during evaluation.

### D.1 Dealing with coexisting emotions

**Model's Emotion Prediction in Conversation:**
- **Examples**: demonstrates how model deals with coexisting emotions during a conversation
- **First prompt**: target emotion is 'excited', predicted emotion is 'anticipating' but still appropriate
- **Response**: model predicts similar response as target, semantically the same
- **Subsequent prompts**: emotions are aggregated and dominant emotion becomes prominent (in this case, 'excited')
- **Emotion prediction accuracy**: model can mimic human empathy by imitating emotions, making conversations more relatable.

**Conversation Example:**
- Speaker expresses excitement about attending a concert
- Model reacts with excitement as well, showing understanding of the speaker's feelings
- Occasional empathic response from model is natural and helps build rapport between conversational partners.

### D.2 Dealing with different intensities of the same emotion

The model's response is more empathetic than the target. It correctly identifies the prompter as terrified and responds accordingly, acknowledging their implied fear: 

Input Prompt: "I'm so scared of bugs! I found one in my hair yesterday and almost died."

Target Response: "ok"

Predicted Response: "Oh no! That's so scary! What kind of bug was it?"

### D.3 Dealing with uncertainty

**Assessing Emotional State: Uncertainty and Model Predictions**

**Example of Ambiguous Prompt**
- Single prompt may not provide enough context for accurate emotional assessment
- Emotional state may be unclear or ambiguous, leading to uncertainty in model predictions

**Case Study: Chik-Fil-A Conversation**
1. **Speaker's Initial Prompt**: Had a craving for Chik-Fil-A
2. **Predicted Emotions**: (disappointed: 2, content: 2, anticipating: 2, jealous: 1, disgusted: 2, hopeful: 1) → disappointed
   - Model unable to determine exact emotion due to ambiguous prompt
3. **Follow-up Prompt**: Realized it was Sunday and parking lot was empty
4. **Revised Target Emotion**: Disappointed about not getting desired food
5. **Predicted Emotions** (disappointed: 11, content: 2, anticipating: 2, jealous: 1, disgusted: 2, hopeful: 1, sad: 1) → disappointed
   - Correct emotion identified with high weight in predictions

**Implications of Uncertainty and Model Predictions:**
- Multiple emotions predicted for ambiguous prompts
- First emotion listed in case of a tie is selected as final prediction.

## Appendix E Reddit’s Subreddits

**Reddit Communities Related to Suicide Risk Factors**

**r/suicidewatch:**
- Support forum for individuals experiencing suicidal thoughts or concerned about others
- Characterized by disproportionate frequency of extremely negative emotions like 'devastated', 'sad', 'lonely', and 'afraid'

**r/depression:**
- Supportive community for people struggling with depression
- Emotional profile: disproportionate frequency of extremely negative emotions, lack of positive feelings
- Most prominent characteristical emotions are 'sad', 'lonely', 'devastated', and 'ashamed'

**r/bpd:**
- Focuses on Borderline Personality Disorder (BPD)
- Emotional embedding: varied across the emotional spectrum, most prominent emotions are 'lonely', 'devastated', 'apprehensive', and 'anxious'

**r/addiction:**
- Community dedicated to discussing addiction
- Related risk factor for suicide
- Similar emotional profiles as r/alcoholism
- High frequencies of 'ashamed' and 'apprehensive'

**r/schizophrenia:**
- Dedicated to individuals with schizophrenia, a primary psychotic disorder
- Emotional embedding: consistent emotions across the two communities, high frequencies of 'anxious', 'afraid', and 'terrified'

**r/ptsd:**
- Space for individuals suffering from PTSD, a disorder that usually arises after experiencing or witnessing a traumatic event
- Emotional embedding: similar to schizophrenia, high frequencies of 'anxious', 'afraid', and 'terrified'

**r/bipolarreddit:**
- Dedicated to discussions about bipolar disorder
- Characterized by extreme shifts in mood, energy, and activity levels
- Emotional profile: characterized by 'apprehensive' and 'anxious' feelings

**r/socialanxiety, r/anxiety and r/healthanxiety:**
- Various subreddits related to anxiety
- Emotions expressed are dominated by 'anxiety'
- Differences in frequency of other negative emotions (lonely for social anxiety, afraid and terrified for health anxiety)

**r/lonely:**
- Community for those feeling loneliness or isolation
- Consistent detection of 'lonely' emotion

**r/adhd:**
- Centered around Attention Deficit Hyperactivity Disorder

**r/autism:**
- A community for those affected by autism.

