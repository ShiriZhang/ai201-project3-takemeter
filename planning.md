# TakeMeter — Planning Document
## Project 3 · AI201 · r/soccer

---

## 1. Community

r/soccer is one of the largest general football communities on Reddit. Posts span a wide
range of content types — match statistics, transfer news, player quotes, media opinion
pieces, and match clips — making discourse quality highly variable. The community is
highly active and all posts are publicly accessible, making it straightforward to collect
200+ labeled examples. The `factual` vs. `opinion` distinction is meaningful in this
community: r/soccer users routinely judge whether a post is sharing information or
expressing a stance, and that judgment affects how they engage with it.

---

## 2. Labels

### `factual`
A post whose primary purpose is to share objective information — including statistics,
match news, player quotes, or match clips — where the core content does not contain
an evaluation or prediction about a person, team, or event made by the poster.

**Example 1:**
> "Norway's Erling Haaland is the first player to score twice on their World Cup debut
> since Elijah Just for New Zealand on Monday, who was the first player to achieve the
> feat since Yasin Ayari for Sweden on Sunday..."

**Example 2:**
> "Cristiano Ronaldo has gone 10 consecutive World Cup and European Championship
> matches without scoring"

---

### `opinion`
A post whose core content contains an evaluation or prediction about a player, team,
match, or football topic — regardless of whether that evaluation comes from the poster
themselves, a person being quoted, or a media article being shared.

**Example 1:**
> "The Athletic on Alexi Lalas: He is one of the most insufferable analysts in American
> TV sports history. And while he was one of the best all-time U.S. defenders, his
> credentials are paltry compared with those of Zlatan and Henry."

**Example 2:**
> "10 men and a statue: Portugal are sacrificing another World Cup for Cristiano
> Ronaldo's ego"

---

## 3. Hard Edge Cases

### Edge Case 1: Statistics with an Implied Stance

**Ambiguous post:**
> "Cristiano Ronaldo has gone 10 consecutive World Cup and European Championship
> matches without scoring"

This statistic was clearly chosen to imply that Ronaldo underperforms in major
tournaments, but the text itself contains no evaluative language.

**Decision rule:** As long as the body of the post consists of verifiable, objective
information, label it `factual` — regardless of the poster's motivation for sharing it.
The classifier reads text, not intent.

---

### Edge Case 2: Quote Posts

**Ambiguous post:**
> "Materazzi: Ibrahimovic is the greatest Inter fan in history for what he's doing
> at Milan"

The poster is only relaying a quote, but the quoted content itself is a strong evaluative
judgment.

**Decision rule:** Assess whether the content of the quote is an independently
verifiable objective fact.

| Quote type | Example | Label |
|-----------|---------|-------|
| States an objective fact | "Canada Soccer: Koné has had successful surgery" | `factual` |
| States a personal decision or intention | "Neuer: This is my last tournament" | `opinion` |
| Evaluates a person | "Materazzi: Ibrahimovic is the greatest Inter fan" | `opinion` |
| Makes a prediction | "Pochettino: Our players will be the great heroes" | `opinion` |

---

### Edge Case 3: Shared Media Articles

**Ambiguous post:**
> "The Athletic on Alexi Lalas: He is one of the most insufferable analysts..."

The poster is only sharing a link; the judgment comes from the media outlet.

**Decision rule:** Label based on the core content being delivered, not on who is
delivering it. If the content is an evaluation or prediction → `opinion`. If the content
is objective fact → `factual`.

---

## 4. Data Collection Plan

Data will be collected from r/soccer using Reddit's flair system as a filter:

- **`factual`**: primarily from posts tagged Stats, News, Quotes, and Media flairs
- **`opinion`**: primarily from posts tagged Opinion Piece, and from user-authored
  posts with clearly evaluative titles

Target: **100 examples per label, 200 total.**

If `opinion` examples fall short after browsing Hot and Top → This Week:
1. Expand to Top → This Month and Top → This Year to source older posts
2. If still insufficient, broaden the search to user-authored posts with explicitly
   evaluative language in the title (e.g., "X is the best/worst...",
   "Y deserves/doesn't deserve...")

Note: Because data is being collected during the 2026 World Cup, `factual` posts
(Stats, News) will be abundant. Extra effort may be needed to source enough
`opinion` examples if the Opinion Piece flair is underrepresented during this period.

---

## 5. Evaluation Metrics

**Metrics used: Overall Accuracy + Per-class F1 + Confusion Matrix**

Accuracy alone is insufficient. If the two label classes are imbalanced, a model that
always predicts the majority class can achieve high accuracy while having zero ability
to identify the minority class. Per-class F1 is therefore needed to evaluate the model's
performance on each label separately.

F1 is chosen over Precision or Recall individually because misclassifying an `opinion`
post as `factual` and misclassifying a `factual` post as `opinion` are considered equally
costly for this task. F1, as the harmonic mean of Precision and Recall, reflects both
error types equally.

A confusion matrix will also be reported to show the specific direction of label
confusion between the two classes.

---

## 6. Definition of Success

| Metric | Target |
|--------|--------|
| Overall Accuracy | ≥ 85% |
| `factual` F1 | ≥ 0.85 |
| `opinion` F1 | ≥ 0.80 |
| Minimum acceptable F1 for either class | 0.75 |

The zero-shot baseline (Groq `llama-3.3-70b-versatile`) is expected to correctly
classify clear-cut `factual` and `opinion` posts, but to struggle on edge cases
(quote posts, statistics with implied stances). The fine-tuned model's accuracy
should meaningfully exceed the baseline. A gap of less than 5 percentage points
would be a signal to review label design or training data quality.

If overall accuracy exceeds 95%, this should be treated as a warning sign and
checked for test set leakage or labels that are too easy to distinguish.

---

## 7. AI Tool Plan

### 7.1 Label Stress-Testing
The label definitions and all three edge case decision rules will be provided to Claude,
with a request to generate 10 posts that sit at the boundary between `factual` and
`opinion`. If any generated post cannot be cleanly classified using the existing decision
rules, the definitions will be revised before annotation of the 200-example dataset
begins.

### 7.2 Annotation Assistance
Claude will be used to pre-label a batch of unlabeled posts, given the label definitions
and decision rules as a prompt, with instructions to output one label per post.
Pre-labeled results are treated as a draft only — every single example must be
manually reviewed and confirmed before being added to the dataset. Skimming
without genuine review is explicitly avoided. All AI-pre-labeled examples will be
marked in the CSV's `notes` column as `"pre-labeled by AI, human reviewed"` and
disclosed in the README's AI Usage section.

### 7.3 Failure Analysis
After training, the full list of misclassified test examples will be provided to Claude
with a request to identify systematic error patterns — for example, quote posts being
consistently mislabeled, or short posts being harder to classify. Claude's suggested
patterns will be verified manually by re-reading the relevant examples before any
pattern is reported in the evaluation section of the README.

---

*This document was written before any data collection began, in accordance with
the Milestone 2 requirements of the TakeMeter project spec.*