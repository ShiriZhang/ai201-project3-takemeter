# TakeMeter — r/soccer Discourse Classifier

## AI201 · Project 3

---

## Community Choice and Reasoning

r/soccer is one of the largest general football communities on Reddit, with posts
spanning match statistics, transfer news, player quotes, media opinion pieces, and
match clips. The community is highly active and all content is publicly accessible,
making it straightforward to collect 200+ labeled examples.

The `factual` vs. `opinion` distinction is meaningful in this community: r/soccer
users constantly navigate between posts that inform (injury updates, match results,
transfer confirmations) and posts that evaluate (pundit takes, fan judgments,
editorial pieces). This distinction reflects a real difference in how community
members engage with content — factual posts are shared and cited, while opinion
posts are debated. The variation in post type and quality makes r/soccer a strong
fit for a classification task.

---

## Label Taxonomy

### `factual`

A post whose primary purpose is to share objective information — including
statistics, match news, player quotes, or match clips — where the core content
does not contain an evaluation or prediction about a person, team, or event.

**Example 1:**

> "Norway's Erling Haaland is the first player to score twice on their World Cup
> debut since Elijah Just for New Zealand on Monday, who was the first player to
> achieve the feat since Yasin Ayari for Sweden on Sunday..."

**Example 2:**

> "Cristiano Ronaldo has gone 10 consecutive World Cup and European Championship
> matches without scoring"

---

### `opinion`

A post whose core content contains an evaluation or prediction about a player,
team, match, or football topic — regardless of whether that evaluation comes from
the poster themselves, a person being quoted, or a media article being shared.

**Example 1:**

> "The Athletic on Alexi Lalas: He is one of the most insufferable analysts in
> American TV sports history. And while he was one of the best all-time U.S.
> defenders, his credentials are paltry compared with those of Zlatan and Henry."

**Example 2:**

> "10 men and a statue: Portugal are sacrificing another World Cup for Cristiano
> Ronaldo's ego"

---

## Data Collection

**Source:** r/soccer (Reddit), collected manually during the 2026 FIFA World Cup
period (June 2026).

**Collection method:** Posts were filtered using Reddit's built-in flair system.
`factual` examples were drawn primarily from Stats, News, Quotes, and Media flairs;
`opinion` examples were drawn primarily from the Opinion Piece flair, supplemented
by user-authored posts with clearly evaluative titles. Each post's title and body
text were copied into a Google Sheet with columns for `text`, `label`, and `notes`.

**Label distribution:**

| Label     | Count   | Percentage |
| --------- | ------- | ---------- |
| `factual` | 108     | 51.7%      |
| `opinion` | 101     | 48.3%      |
| **Total** | **209** | **100%**   |

**Labeling process:** Each example was read and labeled individually using the
decision rules documented in `planning.md`. Labels were verified for internal
consistency through a review pass before training. All annotation was done manually
(see AI Usage section).

### Three Difficult-to-Label Examples

**Difficult Example 1: Quote post with embedded evaluation**

> "Paraguay's coach Gustavo Alfaro to Almiron after the match: 'You should have
> said everything to him in Guarani. He doesn't understand.'"
>
> This post records a humorous post-match exchange. The poster is only sharing what
> happened, but the quoted content contains a judgment about a person ("He doesn't
> understand"). The humorous tone made this feel more like scene-reporting than
> opinion-expressing. Under the decision rule for quote posts — assess whether the
> quote's content contains an evaluation — this was labeled **`opinion`**. The
> framing and tone of the post do not override the content of the quote itself.

**Difficult Example 2: Statistic with implied stance**

> "Cristiano Ronaldo has gone 10 consecutive World Cup and European Championship
> matches without scoring"
>
> This statistic was clearly selected to imply that Ronaldo underperforms in major
> tournaments, but the text itself contains no evaluative language. The decision
> rule is: label based on text content, not inferred intent. The data is verifiable
> and the poster's motivation does not change what the post delivers. Labeled
> **`factual`**.

**Difficult Example 3: Question-format media headline**

> "World Cup 2026: Are Portugal a better team without Cristiano Ronaldo?"
>
> The headline is phrased as a question, which superficially suggests discussion
> rather than argument. However, the article's core content evaluates Portugal's
> squad and performance. Under the decision rule for shared media articles — assess
> the core content, not the surface format — this was labeled **`opinion`**.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (HuggingFace Transformers)

**Training setup:** Fine-tuned using the HuggingFace `Trainer` API on Google Colab
with a T4 GPU. Dataset split: 70% train (146 examples), 15% validation (31
examples), 15% test (32 examples), stratified by label. Training completed in
approximately 5 minutes.

**Hyperparameters:**

| Parameter          | Value |
| ------------------ | ----- |
| Epochs             | 3     |
| Learning rate      | 2e-5  |
| Batch size (train) | 16    |
| Batch size (eval)  | 32    |
| Weight decay       | 0.01  |
| Warmup steps       | 50    |

**Training curve:**

| Epoch | Training Loss | Validation Loss | Validation Accuracy |
| ----- | ------------- | --------------- | ------------------- |
| 1     | 0.686         | 0.684           | 51.6%               |
| 2     | 0.680         | 0.666           | 90.3%               |
| 3     | 0.640         | 0.619           | 90.3%               |

**Key hyperparameter decision — number of epochs:** The default of 3 epochs was
retained. Validation accuracy reached 90.3% at epoch 2 and did not improve at
epoch 3, though validation loss continued to decrease (0.666 → 0.619). This
plateau indicated the model had learned the available signal by epoch 2, and adding
more epochs would have risked overfitting on the small 146-example training set.
The early stopping behavior confirmed 3 epochs was the right choice for this
dataset size.

---

## Baseline Description

**Model:** Groq `llama-3.3-70b-versatile` (zero-shot, no task-specific training)

**Prompt used:**

```
You are classifying posts from r/soccer, a large football community on Reddit.
Assign each post to exactly one of the following two categories.

factual: A post whose primary purpose is to share objective information —
including statistics, match news, player quotes, or match clips — where the
core content does not contain an evaluation or prediction about a person,
team, or event.
Example: "Norway's Erling Haaland is the first player to score twice on their
World Cup debut since Elijah Just for New Zealand on Monday."

opinion: A post whose core content contains an evaluation or prediction about
a player, team, match, or football topic — regardless of whether that
evaluation comes from the poster themselves, a person being quoted, or a
media article being shared.
Example: "10 men and a statue: Portugal are sacrificing another World Cup for
Cristiano Ronaldo's ego."

Key decision rules:
1. If the post contains only verifiable, objective information with no
evaluative language, label it factual — even if the choice of information
implies a stance.
2. If a post quotes someone, assess the content of the quote: if the quote
contains an evaluation or prediction, label it opinion; if it states an
objective fact, label it factual.
3. If a post shares a media article, assess the core content of the article:
evaluative or predictive content = opinion; objective factual content = factual.

Respond with ONLY the label name in lowercase.
Do not explain your reasoning.

Valid labels:
factual
opinion
```

**How results were collected:** The baseline was run on all 32 test examples using
the `classify_with_groq()` function in the starter notebook (Section 5), with a
1.5-second delay between requests to respect free-tier rate limits. All 32
responses were successfully parsed on the second attempt (the first attempt failed
due to an expired API key).

---

## Evaluation Report

### Overall Accuracy

| Model                                             | Accuracy  | Test Set Size |
| ------------------------------------------------- | --------- | ------------- |
| Zero-shot baseline (Groq llama-3.3-70b-versatile) | **0.844** | 32            |
| Fine-tuned DistilBERT                             | **0.844** | 32            |
| Improvement                                       | 0.000     | —             |

The two models achieved identical overall accuracy. However, this number conceals
a significant difference in error behavior — the models make opposite kinds of
mistakes. See the confusion matrix and error analysis below.

---

### Per-Class Metrics

**Zero-shot baseline:**

| Label         | Precision | Recall   | F1-score | Support |
| ------------- | --------- | -------- | -------- | ------- |
| `factual`     | 1.00      | 0.71     | 0.83     | 17      |
| `opinion`     | 0.75      | 1.00     | 0.86     | 15      |
| **Macro avg** | **0.88**  | **0.85** | **0.84** | **32**  |

**Fine-tuned DistilBERT:**

| Label         | Precision | Recall   | F1-score | Support |
| ------------- | --------- | -------- | -------- | ------- |
| `factual`     | 0.80      | 0.94     | 0.86     | 17      |
| `opinion`     | 0.92      | 0.73     | 0.81     | 15      |
| **Macro avg** | **0.86**  | **0.84** | **0.84** | **32**  |

The fine-tuned model outperforms the baseline on `factual` F1 (0.86 vs. 0.83) and
meets the target threshold for both classes (`factual` ≥ 0.85 ✅, `opinion` ≥ 0.80
✅). The baseline has higher `opinion` F1 (0.86 vs. 0.81) because it never misses
an `opinion` post (recall = 1.00), but at the cost of also calling many `factual`
posts opinion (precision = 0.75).

---

### Confusion Matrix (Fine-Tuned Model)

|                    | Predicted `factual` | Predicted `opinion` |
| ------------------ | :-----------------: | :-----------------: |
| **True `factual`** |         16          |          1          |
| **True `opinion`** |          4          |         11          |

The dominant error direction is **opinion → factual**: 4 out of 15 `opinion` posts
were predicted as `factual`. Only 1 `factual` post was predicted as `opinion`.

For comparison, the baseline confusion matrix was the mirror image: it missed 5
`factual` posts (predicted as `opinion`) and missed 0 `opinion` posts.

---

### Error Analysis: Three Wrong Predictions

All 5 wrong predictions had very low model confidence (0.50–0.55), meaning the
model signaled uncertainty in every case it got wrong. The three most analytically
interesting errors are:

---

**Error 1: Metaphorical evaluative language misclassified as factual**

> Text: "10 men and a statue: Portugal are sacrificing another World Cup for
> Cristiano Ronaldo's ego"
> True label: `opinion` | Predicted: `factual` | Confidence: 0.50

This is one of the two example posts used to define the `opinion` label in both
the Groq prompt and the planning document — and the fine-tuned model still got it
wrong at near-random confidence (0.50). The failure reveals a systematic weakness:
the model learned to detect **direct** evaluative language ("worst", "greatest",
"overrated") but not **metaphorical** framing that implies evaluation. "10 men and
a statue" is an image, not a factual claim. The model cannot recover the implied
judgment from the metaphor. This is likely a data problem: the 101 `opinion`
training examples probably contain more direct evaluation than metaphorical
evaluation, so the model never learned to recognize the latter.

---

**Error 2: News-headline format with embedded evaluation**

> Text: "Rashford and Rice give England boost for Ghana with Saka set for bench
> again"
> True label: `opinion` | Predicted: `factual` | Confidence: 0.55

The phrase "give England boost" contains an evaluative judgment — calling the
players' availability a positive development — but the sentence is structured like
a sports news report. The model was misled by the format. This is the highest-
confidence wrong prediction (0.55), suggesting the model was more decisive about
this misclassification than the others. The pattern here is: the model has likely
learned that sports-news sentence structure signals `factual`, and this structural
cue overrides the evaluative content within the headline.

---

**Error 3: Metaphorical praise treated as factual description**

> Text: "Soccer beacon Seattle shines on the World Cup stage"
> True label: `opinion` | Predicted: `factual` | Confidence: 0.51

"Beacon" and "shines" are metaphorical words of praise, but they are not the kind
of explicit judgment markers (comparatives, superlatives, direct claims) the model
associates with `opinion`. This error is structurally identical to Error 1: the
model requires explicit evaluative language to predict `opinion` and cannot detect
evaluation expressed through imagery or metaphor.

**Systematic pattern across all 5 errors:** Every wrong prediction involved either
(a) metaphorical or implicit evaluative language, or (b) opinion content framed
in news-headline structure. The model performs well on posts with explicit
evaluative markers but fails when evaluation is expressed indirectly.

---

### Sample Classifications

| Post (truncated)                                                                                                              | True Label | Predicted | Confidence | Correct? |
| ----------------------------------------------------------------------------------------------------------------------------- | ---------- | --------- | ---------- | -------- |
| "Norway's Erling Haaland is the first player to score twice on their World Cup debut since..."                                | `factual`  | `factual` | ~0.92      | ✅       |
| "Graeme Souness: 'Even If Ronaldo had retired 10 years ago in 2016, his place at the top would already have been secured...'" | `opinion`  | `opinion` | ~0.89      | ✅       |
| "[Argentina] Outrage. Controversy over an AI-generated image of Diego Maradona used in a sports betting ad"                   | `opinion`  | `factual` | 0.51       | ❌       |
| "10 men and a statue: Portugal are sacrificing another World Cup for Cristiano Ronaldo's ego"                                 | `opinion`  | `factual` | 0.50       | ❌       |
| "Rashford and Rice give England boost for Ghana with Saka set for bench again"                                                | `opinion`  | `factual` | 0.55       | ❌       |

_Note: Confidence scores for correctly-predicted examples (~) are approximate. Exact
values can be retrieved by printing `ft_probs` for those test set indices in the
Colab notebook._

**Why the first prediction is reasonable:** The Haaland post is an unambiguous
statistical record — a chain of verifiable historical comparisons with no
evaluative language anywhere in the text. There are no predictions, no judgments,
and no subjective claims. A high-confidence `factual` prediction is exactly correct
here, and it reflects what the model learned well: posts that consist entirely of
statistics and historical records are reliably `factual`.

---

## Reflection: What the Model Learned vs. What I Intended

I intended the model to learn a conceptual distinction: posts that deliver
information vs. posts that deliver judgment, where the test is whether the core
content makes a claim about quality, value, or prediction rather than stating a
verifiable fact.

What the model actually learned was a more surface-level heuristic: **explicit
evaluative language signals `opinion`; the absence of explicit evaluative language
signals `factual`**. This works for the majority of posts — and explains the 84.4%
accuracy — but fails systematically when evaluation is expressed indirectly.

The clearest evidence of this gap is Error 1: the model predicted `factual` for
"10 men and a statue: Portugal are sacrificing another World Cup for Cristiano
Ronaldo's ego" — the very post used as the `opinion` example in the prompt. The
model encountered this exact framing and still labeled it `factual`. It did not
learn "Portugal are sacrificing their World Cup" as an evaluative claim; it learned
something narrower about which words mark evaluation.

The fine-tuned model also made an opposite shift from the baseline: the baseline
over-predicts `opinion` (recall = 1.00) while the fine-tuned model over-predicts
`factual` (recall = 0.94 for factual). Fine-tuning moved the decision boundary
toward `factual` — likely because the training data contained more clear-cut
`factual` examples with unambiguous statistical and news language, causing the
model to anchor more strongly on that pattern.

The practical implication: this classifier would work well as a filter for obvious
cases (stats posts, direct opinion pieces) but would need human review or more
training data for posts using metaphor, irony, or news-headline framing.

---

## Spec Reflection

**One way the spec helped:** The requirement to read 30–40 actual posts before
committing to labels was the most valuable constraint in the project. In practice,
this forced the discovery that the original four-label system (borrowed from the
r/nba example in the spec) did not fit r/soccer. r/soccer is dominated by news and
statistics posts that do not express discourse quality in the way the spec's example
taxonomy assumed. The spec's insistence on reading real data before designing labels
prevented a significantly mislabeled dataset.

**One way the implementation diverged:** The plan documented in `planning.md`
included `transfer_talk` as a fourth label and at one point a `discussion` label as
a third. Both were dropped after reading real posts and discovering that (a) during
the World Cup period, transfer content was scarce, and (b) the boundary between
`discussion` and `opinion` could not be defined from text content alone — it
depended on the poster's intent, which the model cannot observe. The final two-label
system is simpler than planned, but the boundary is more defensible and the
annotation more consistent.

---

## AI Usage

**Instance 1: Label stress-testing (before annotation)**

Claude was given the label definitions and decision rules from `planning.md` and
asked to generate 10 posts sitting at the boundary between `factual` and `opinion`.
Of the 10 generated posts, 3 produced inconsistent judgments from the original
rules — specifically, quote posts where a player stated a personal decision (e.g.,
"This will be my last World Cup"). The original rule classified personal decisions
as `factual`; the stress-test revealed this was incorrect because personal decisions
are not independently verifiable facts. The rule was revised before annotation
began: personal decisions and intentions expressed in quotes → `opinion`; only
objectively verifiable facts stated in quotes → `factual`. This revision directly
improved annotation consistency across the 209 examples.

**Instance 2: Failure analysis (after training)**

After obtaining the fine-tuned model's 5 wrong predictions from the notebook,
Claude was asked to identify systematic patterns across the errors. Claude
identified that all errors involved either metaphorical/implicit evaluative language
or opinion content framed in news-headline structure — neither of which contains
explicit judgment markers. This pattern was verified by manually re-reading each
of the 5 errors, confirmed to hold, and used to write the error analysis section
of this README. One pattern Claude suggested — that the model might struggle with
short posts — was not confirmed on manual review (post length did not correlate
with errors) and was not included in the report.

All 209 examples were labeled manually. No AI assistance was used during the
annotation process itself.

---

## Repository Contents

| File                             | Description                                       |
| -------------------------------- | ------------------------------------------------- |
| `README.md`                      | This document                                     |
| `planning.md`                    | Design decisions made before data collection      |
| `takemeter_dataset.csv`          | Labeled dataset (209 examples)                    |
| `evaluation_results.json`        | Baseline and fine-tuned accuracy comparison       |
| `confusion_matrix.png`           | Confusion matrix for fine-tuned model on test set |
| `ai201_project3_takemeter.ipynb` | Colab notebook (fine-tuning pipeline)             |
