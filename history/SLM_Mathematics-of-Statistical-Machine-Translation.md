# The Mathematics of Statistical Machine Translation: Parameter Estimation

## Introduction
- Growing availability of bilingual, machine-readable texts has stimulated interest in extracting linguistically valuable information from such texts.
- Recent papers deal with the problem of automatically obtaining aligned sentence pairs from parallel corpora (e.g., Warwick and Russell 1990; Brown, Lai, and Mercer 1991; Gale and Church 1991b; Kay 1991).
- Simple statistical methods can be surprisingly successful in achieving linguistically interesting goals.

**Word Alignment of Sentence Pairs**:
- The authors propose a statistical approach to machine translation from French to English (Brown et al., 1988, 1990).
- They sketch an algorithm for estimating the probability that an English word will be translated into any particular French word.
- These probabilities can then be used together with a statistical model of the translation process to align words in an English sentence with those in its French translation.
- Pairs of sentences with aligned words offer a valuable resource for work in bilingual lexicography and machine translation.

**Approach Outline**:
- **Section 2**: A synopsis of the statistical approach to machine translation, following word alignment of pairs of sentences.
- **Section 4**: Description of a series of models of the translation process, along with algorithms for estimating their parameters from data.
- **Section 5**: Presentation of results obtained by estimating the parameters for these models from a large collection of aligned sentence pairs from the Canadian Hansard data (Brown, Lai, and Mercer 1991).
- **Section 6**: Discussion of shortcomings of the models and proposals to address them.
- **Final Section**: Discussion of the significance of the work and possibilities for extending it to other language pairs.

## 2. Statistical Translation

**Statistical Translation Approach**
- In 1949, Warren Weaver proposed using statistical methods for machine translation (Weaver 1955)
- Early efforts abandoned due to philosophical and theoretical reasons
- With modern computing power, statistical approach is viable

**Fundamental Equation of Machine Translation**
- Assign probability Pr(f|e) to every pair of strings: English (e), French (f)
- Probability represents translator's likelihood of producing a specific translation
- Objective: Find English string e with highest probability Pr(e|f) = argmax Pr(e) Pr(f|e)

**Challenges in Statistical Translation**
1. **Language Modeling Problem**: Estimating Pr(e)
2. **Translation Modeling Problem**: Estimating Pr(f|e)
3. **Search Problem**: Finding optimal English string

**Importance of Well-Formed Strings**
- Distinguish well-formed from ill-formed strings
- Important for translation model to focus probability on well-formed English strings
- Reversing models would not achieve this as they are prodigal with probability, especially on ill-formed strings.

## 3. Alignments

**Alignments:**
- Translation vs Alignment: translation - strings that are translations of one another (enclosed in parentheses), alignment shows connection between words in different languages
- Graphic representation using lines or connections between related words
- One English word can connect to multiple French words, forming more complex alignments
- Each alignment is correct with some probability, less probable than the most straightforward one
- Set of English words connected to a French word called "cept" generating that French word in an alignment.
- Empty cept may be included if no English words are connected to certain French ones (e.g., articles or conjunctions)
- Formal definition: subset of positions in the English string and their corresponding words
- Denoted as A(e,f), where e is an English passage and f is its French translation
- Number of alignments depends on length of both strings, with 2lm possible connections between them.

## 4. Translation Models

**Translation Models: Overview and Likelihood Computation**

**Model Development:**
- Series of five translation models with algorithms for computing conditional probability Pr(f|e), or likelihood of a translation (f,e)
- Prescription for computing Pr(f|e) as a function of free parameters to be estimated through training process called EM algorithm
- Models progressively more complex: Models 3, 4, and 5 include connections between words based on their identities and positions in strings

**Training:**
- Estimate parameters by iteratively approaching local maximum of likelihood of a set of translations (training data) using EM algorithm
- Initial guess for parameters can influence the approach to the maximum if there are multiple local maxima
- Models 1 and 2 have unique local maximums, allowing explicit computation of sums over alignments and derivation of parameter estimates without dependence on initial estimates.

**Mathematical Object:**
- Joint probability distribution Pr(F = f, A = a, E = e) where F and E are French and English strings respectively, and A is an alignment between them.
- Likelihood of (f | e) = Σ_a Pr(f, a | e)

**Assumptions:**
- No more than one connection per word in the English string or it is empty
- If the word in position j of the French string is connected to the word in position i of the English string: aj = i; otherwise aj = 0.

**Alignment Representation:**
- Represented by a series, a = a1 a2 ... am, of m values, each between 0 and 1 where ai represents the position in the French string connected to the word at position i in the English string.

**Probability Distribution:**
- Pr(f,ale) = Pr(m|e) ∏j=1m Pr(aj|a1,...,aj-1,fj-1,m,e)
- Equation (4) is not an approximation; it can always be analyzed into a product of terms representing choices made when generating a French string and its alignment from an English one.

### 4.1 Model 1

**Model 1 Assumptions:**
- Conditional probabilities not all independent: Pr(m|e) assumed to be constant and unnormalized (minor issue)
- Translation probabilities depend only on fj, aj, and m
  - Assume small, fixed number for € = Pr(m|e)
  - t(fj|aj) is the translation probability of fj given aj

**Joint Likelihood:**
- Given a French string and an alignment: Pr(f, ale) = (1 + 1)m ∏j=1m t(f|ej)
- Alignment determined by specifying values for aj

**Maximizing Translation Probabilities:**
- Introduce Lagrange multipliers λe to find unconstrained extremum of h(t, λ)
- Extremum occurs when partial derivatives with respect to t and λ are zero
- Equation (10) suggests an iterative process called the EM algorithm for finding a solution

**Expected Number of Connections:**
- c(f|e; f, e) = expected number of times e connects to f in translation
- Replace Ae by Ae Pr(fle) and use Equation (13) instead of Equation (11) for practical implementation
- In practice, calculate counts efficiently using Equation (15) instead of Equation (12)

### 4.2 Model 2

**Model 1 vs Model 2**

**Model 1**:
- No consideration of word order within strings
- First word in French string equally likely connected to any English word

**Model 2**:
- Same assumptions as Model 1, but with additional assumptions:
  - **Alignment probabilities**: a(a_j | j, m, I) = Pr(a_j | a_{j-1}, f_{i-1}, m, I)
    - Constrained to sum up to 1 for each triple (j, m, l)
  - Equation (6) becomes: **Pr(f | e)** = **∑ t(f | e) a(a_j | j, m, I)**

**New Terms**:
- **Alignment probabilities**: a(a_j | j, m, I) = Pr(a_j | a_{j-1}, f_{i-1}, m, I)
- **c(ilj, m, l; f, e)**: expected number of connections between words in positions i and j of strings f and e

**Equations**:
- Equations (12), (13), and (14) carry over from Model 1 to Model 2 unchanged
- New equation: **Pr(f | e)** = **∑j=1l ∑i=0t(f|e) a(ilj, m, 4)**
- Equation (27): c(f|e; f, e) = **∑m t(f|e) a(ilj, m, 1)** + **∑l t(f|e) a(ilj, m, 4)**

**Transitioning from Model 1 to Model 2**:
- Equivalent to computing probabilities of all alignments as if dealing with Model 1
- Then collecting counts in a way appropriate to Model 2

### 4.3 Intermodel Interlude

**Intermodel Interlude**
- **Equation (4)** is one way to write joint likelihood of f and a as product of conditional probabilities
- Each such product corresponds to a generative process for developing f and a from e
- In Model 1:
  - Decide length of f, then connect position in e to each part of f
  - Determines distribution Pr(d0) for random variable Ce (fertility of e)
- In Models 3, 4, and 5:
  - Write joint likelihood as product of conditional probabilities differently
  - Choose fertility and list of French words to connect to each English word (tablet)
  - Permute tablets to produce f
  - Random variables: T (tableau), π (permutation), lik (position in f)
- Joint likelihood for tableau, permutation:
  ```|Pr(v, π | e)|Pr(oi | oi-1, e)|Pr(po | oi, e)| |---|---|---| |∏ Pr(¢i | ¢i-1, e)|∏ Pr(Tik | vi-1, πi-1, T0, e)|∏ Pr(T0 | v0, π0, T0, e)|∏ Pr(Tk | v0, π0, T0, e)|```
- Pr(f, a | e) = ∑(T, π) Pr(v, T | e)
- Viterbi alignment for (f,a):
  - Most probable alignment among those leading to pair f,a
  - Not necessarily unique; denote set by (f,a)
  - For Model 2, finding Vi(fe) is straightforward through choosing Qj to maximize product of t and a
- Viterbi alignment depends on model, denoted as Vi(fe; i), where i is the model number.

### 4.4 Model 3

**Model 3**

**Assumptions**:
- Pr(oiloi-1,e) depends only on Di and e/
- Pr(Tixtvlf-1, Tixlvf -1, m, e) depends only on Ti, ei
- Pr(¢i[¢~-1, e) depends only on ¢i and ei
- Pr(Tixlvf-1, 7i-1, 0.8,e) depends only on Ti, i, m, and /

**Parameters**:
- Fertility probabilities: n(ole) = Pr(¢[ e/)
- Translation probabilities: t(fle) = Pr(Tik = fjvik-1 7i-1, percent,1~e)
- Distortion probabilities: d(jli, m,4) = Pr(Ilik = j[w/k-l, Tij -1, TOij 66,46,e)

**Distortion Probability for the Empty Cept**:
- Contribution is 1/(40 - k), where k is the number of words in the string
- Expects words to be spread uniformly throughout the French string

**Length Dependence**:
- Pr(p0lo5,e) depends on the length of the French string
- Longer strings should have more extraneous words

**Likelihood Calculation**:
- h(t,d,n,p,A,/,V, €) = Pr(fle) - Pr(fle) EA(gI(le) - 1) - rin(Edlli,, m,l)- 1)Ae(~/t(fle - 1)- ~-~#im,(~.d(/li,) m,4) - 1) ve(En(ole) - 1) - E(po + pl - 1).i
- The function h is used to find the constrained minimum likelihood of a translation with Model 3

**Counts**:
- c(fle; f,e) = E Pr(ale,f) C 6(f,f)6(e, e),m
- c(fle;f,e) = ~a Pr(ale,f)y~. 6(S,:)6(e, percent),j=1 j=l
- c(jli,m,1;f,e) = E Pr(ale,f)6(i,0;) ,
- cqli,m, I;f,e) = y~ Pr(ale,f)6(i,aj),
- c(ole; f,e) = C Pr(ale,f) i=1 6(0,01)6(e, e),l
- c(¢ Ie; f, e) = E Pr(ale,f) ~ 6(¢, ¢i)6(e, ei),
- c(0;f,e) = c(O; f, e) = ~[~ Pr(ale , f)(m - 2¢0)EPr(ale,f)(m 200)
- c(1;f,e) = E Pr(ale,f)po,
- c(1; f, e) = ~a Pr(ale, f)¢0.

**Reestimation Formulae**:
- t(fle) = Ael t(fle) = A2 a ~_,
- d(jli, m,1) = Kinl ~ S dO[i, m, l) ~- # i m l--1 s=]
- n(ole) = vel s=1 c(¢1 e; f(s), e(S)),S c(ole; f() , e6)
- n(¢ Ie) = u [ I ~ 5=1 andand pk = &-1 S=1 c(k; f(s)e(511. Pk = ~-1 ~ (42)(42)

### 4.5 Deficiency

**Model 3 Deficiency**
- Problem with parameterization of distortion probabilities: sum over all pairs does not equal unity due to assumption that Pr(I|k z ][T|Fi1 ~ 7o , 06,e) depends only on j,i, m, and l for i > 0
- Model wastes probability on generalized strings with some positions having several words and others having none (deficiency)

### 4.6 Model 4
- Alleviates problem of phrases moving around as units in Model 3
- Modifies treatment of Pr(Ilik) to account for the tendency of phrases to move around as units
- **CEPTUAL SCHEME**:
    - Resolves English string into set of possibly overlapping cepts
    - Each cept accounts for one or more French words
    - One-word cepts have natural order corresponding to position in English string
    - **CENTER @i**: ceiling of average value of positions of words from its tablet
    - **HEAD**: word in its tablet with smallest position in the French string
- **DISPLACEMENT FOR HEAD OF CEPT i**: either positive or negative
    - Expect d1(-1|A(e), B(f)) to be larger than d1(+1|A(e), B(f)) when e is an adjective and f is a noun
- **Subsequent Words of CEPT i**: require that they be placed in order, not necessarily consecutively
    - Only one arrangement of T[i] is possible due to the requirement for subsequent words to lie to the right of previous ones

**Evaluating Adjusted Likelihood with Model 4**
- Complicated by changing centers and contributions of several words when moving a French word between cepts
- Define **b(a)** and **b[i](a)** analogously to Pr(a|e,f;n) for n=3 and n=4 using Model 3 and Model 4 respectively
- Evaluate adjusted likelihood incrementally

### 4.7 Model 5

**Model 5+**

**Deficiencies of Models 3 and 4**:
- In Model 4: Words can lie on top of one another, and words can be placed before the first or beyond the last position in the French string.
- In Model 5: We remove these deficiencies for one-word cepts by placing words correctly after T[i]-1 and T[i]k-1 are placed.

**Model 5 Parameters**:
- Retains **two sets of distortion parameters**, as in Model 4, referred to as **d1** and **d>1**.
- Enforces the constraint that **T[i]k** be placed in one of the remaining vacancies after T[i]-1 and T[i]k-1 are placed.

**Distortion Parameters**:
- For **[i] > 0**, the probability:
  - **Pr(T[i]j = *1"1|lnji-1,78,08,e) -- d1 (v;1B(f), voi-1' Vm ;1~[i]-1 ~1 ¢/, e)**.
    - **v** is the number of vacancies up to and including position j before T[i]k is placed.
  - The remaining probability depends on **Vm**, the number of vacancies remaining in the French string, as well as **P1**, **f**, and **e**.
- For **[i] > 0** and **k > 1**, the probability:
  - **Pr(Tlplk = jltiijlTr[i]~-i ' Tjl-1,78,08,e)k-1 71"1[i]-1'7"6,z~b~,e)** = **d21(;j - vt,lk_ll3(fj),vm - vt,lk_, - ~b[i]+k)(1 - 8(vj, vj_l))**.
    - The remaining probability depends on the final factor enforcing that TiiJk lands in a vacant position.

**Model 5 Evaluation**:
- No incremental evaluation of likelihood is possible due to recomputation requirements.
- Alignments for Model 5 are included in expectation computation, with some trims based on probability thresholds.

**Role of Models 2, 3, and 4**:
- They serve as "weaker but more agile brethren" led by Model 5 in the alignment battle.
- Their parameters are adjusted during EM algorithm iterations using Model 5 probabilities.

## 5. Results

**Model Training Results**

**Training Data**:
- Large collection of training data used to estimate model parameters
- Extracted from parallel corpora of French and English sentences
- Selected only those sentences with less than 30 words in length

**Vocabulary Selection**:
- Chose vocabulary by selecting words that appear at least twice in their respective languages
- Replaced rare or unknown words with "unknown" placeholders

**EM Algorithm Iterations**:
- Performed 12 iterations of the EM algorithm to refine the model
- Initially set all translation probabilities to 1/58,016 (equal probability for each word in the other language)
- Retained only probabilities that surpassed a threshold after each iteration

**Model Comparison**:
- Models with unspecified Pr(mle) assumed Poisson distribution for French string length
- Model 3 had significantly different alignment patterns compared to Models 1 and 2

**Viterbi Alignments**:
- Changed as iterations progressed, becoming more dominant and accurate
- Example sentences showed improvements in connecting words between languages with each iteration

### Figure 5 The progress of alignments with iteration

**Progress of Alignments with Iteration**

**Pseudographic Improprieties**:
- Analyzed des into its constituents: de and les
- Committed orthographic French improprieties to regularize the text

**Translation Probabilities and Fertilities**
- Shown for a selection of English words
- Only probabilities greater than 0.01 are shown
- Some words have multiple translations, leading to large fertility and spread of translation probabilities over various words

**French Language Complexity**:
- French allows saying the same thing in many ways
- Examples: should rarely has a fertility greater than one but still produces many different words, including devrait, devraient, devrions, doit, doivent, devons, and devrais

**Adjectives Behave Better**:
- national almost never produces more than one word and confines itself to nationale, national, nationaux, and nationales (feminine, masculine, masculine plural, feminine plural) of the corresponding French adjective

**Articles in Translation**:
- The indefinite article "the" has a fertility of 1 but sometimes generates different translations depending on which language prefers an article
- Examples: farmers have about 14% of their translation probabilities with no article, while English tends to prefer articles where French does not; conversely, some French words (e.g., "farmers") may require the inclusion of an article in the English translation

**Alignment Analysis**
- Figures 16, 17, and 18 show automatically derived alignments for three translations, demonstrating that they are inherent in the Canadian Hansard data itself
- The algorithms used discovered these alignments from 1,778,620 translations without any explicit linguistic knowledge

**Conclusion**:
- These alignments show the intricacies of translation and how even common words can have multiple translations with varying fertilities.

## 6. Better Translation Models

- Models 1-5 effective for word-by-word translation alignments
- Goal is to achieve better overall translation
- Figures 16, 17, 18 show best alignments from large sets
- Examples of English-French translations provided
- Discusses secretary of state and minister translations
- Mentions starred questions in parliamentary context
- References "Mathematics of Statistical Machine Translation"
- Includes dialogue between Speaker/Orateur
- Alignment examples range from 10^25 to 10^31 possibilities
- Text contains mixed languages and formatting issues

### 6.1 The Truth about Deficiency 

**The Truth about Deficiency**

**Problems with Ignoring Morphological Structure**:
- Dilutes strength of statistical model
- Explains each form of French verb independently

**Problems with Ignoring Multi-word Concepts**:
- Forces false or unsatisfactory account in translations
- Leads to deficient models, either in fact (Models 3 and 4) or in spirit (Models 1, 2, and 5)

**Consequences of Deficiency**:
- Probability of failure is not zero: Pr(failure|e) > 0
- Remainder of probability is concentrated on the event "failure"
- **w(e) + i(e)** < 1, where w(e) = sum of Pr(f|e) over well-formed French strings and i(e) = sum over ill-formed French strings
- This can lead to improperly favoring short English strings when finding e given f

**Potential Solutions**:
- Replace **Pr(f|e)** with c' **Pr(f|e)** for some empirically chosen constant c, but this is only a temporary relief
- Better modeling is the true cure


### 6.2 Viterbi Training
- Evaluating expectations becomes increasingly difficult as models progress from 1 to 5
- Converges when reestimating parameters to make current Viterbi alignments as probable as possible
- Reinterpreted to find most probable alignment among those found, rather than actual one

### 6.3 Multi-Word Concepts
- In Models 1-5, restricted to single-word concepts
- Extending generative process to encompass multi-word concepts
- Need to be discriminating in choosing potential multi-word cepts
- Inspection of Viterbi alignments can reveal translations for given multi-word sequences that differ substantially from compositional ones.

### 6.4 Morphology

**French Verb Forms and Morphology:**
- Each distinct sequence of letters is treated as a word in English, with no kinship between forms of irregular verbs (e.g., to eat vs. manger)
- French verbs have many forms: 41 for irregular verb "devoir" and only 39 for regular verb "manger"
- Only a fraction of these forms appear in the training data (13 out of 39 for manger, 28 out of 39 for parler)
- Rare forms can lead to confusion due to lack of relationship with common forms

**Intended Solution:**
- Use inflectional analysis to make relationships between different forms of a word more evident
- Analyze verbs (e.g., je mange la pêche) into root and suffixes indicating tense or person
- This should reduce the French vocabulary by about 50% and English vocabulary by about 20%, improving model statistics.

## 7. Discussion

**Bilingual Corpus and Model Analysis:**
* Large bilingual corpus discussed by Brown et al. (1988) for extracting lexical correlations
* Automatic extraction of correlations using EM algorithm's first iteration (Model 1)
* Unsatisfactory results, suggested removing established correlations and reprocessing
* Proper way to proceed: carry out more iterations of the EM algorithm for Model 1
* Brown et al. (1990): similar model as Model 3 but unaware of logarithm's unique local maximum or summing over alignments
* Gale and Church (1991a) describe an algorithm like Brown et al. (1988), focusing on simultaneous appearance of words in pairs of sentences
* Correlations between English and French words are pronounced, some speak loudly while others whisper
* Models provide accounts of word-by-word alignment for French and English strings
* Intended to translate French into English but believe the same applies to other languages with sufficient translation rate.

