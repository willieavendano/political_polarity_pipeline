# Manual Labeling Guidelines

## Overview

This document provides guidelines for manually labeling documents for political polarity classification. Manual labels are higher quality than distant supervision (outlet labels) and should be used when possible.

## ⚠️ Important Ethical Considerations

- **Label the CONTENT, not the author**: Focus on what is being said, not who is saying it
- **No demographic inference**: Never infer or record protected characteristics of authors
- **Aggregate use only**: Labels are for building aggregate models, not profiling individuals
- **Document disagreement**: Political ideology is complex; disagreement is expected and valuable

## Label Categories

### 0: Left/Democrat/Liberal

**Characteristics:**
- Supports progressive policies (e.g., expanded social programs, climate action)
- Advocates for economic redistribution and regulation
- Emphasizes collective rights and social justice
- Favors larger role for government in solving problems
- Supports labor unions, worker protections
- Advocates for civil rights expansions

**Examples:**
- "We need universal healthcare to ensure every American has access to medical care"
- "Climate change requires immediate government action and green energy investment"
- "Tax reform should ask the wealthy to pay their fair share"

### 1: Center/Mixed/Unclear

**Characteristics:**
- Presents balanced view of multiple perspectives
- Expresses uncertainty or ambiguity about political issues
- Advocates for bipartisan compromise
- Discusses policy without clear ideological framing
- Mixes left and right positions
- Primarily factual/descriptive without advocacy

**Examples:**
- "Both parties have valid points; compromise is needed"
- "The data shows mixed results; more research is required"
- "This complex issue has tradeoffs that must be carefully weighed"
- Straight news reporting without editorial stance

**Important:** Do NOT label as center just because:
- The tone is moderate or polite
- Multiple viewpoints are mentioned (if one is clearly endorsed)
- The topic is controversial

### 2: Right/Republican/Conservative

**Characteristics:**
- Supports free market capitalism and limited government
- Advocates for lower taxes and reduced regulation
- Emphasizes individual responsibility and liberty
- Favors traditional social values and institutions
- Supports strong national defense and border security
- Advocates for Second Amendment rights

**Examples:**
- "Free markets and entrepreneurship drive prosperity better than government programs"
- "Lower taxes and reduced regulations create jobs and economic growth"
- "Traditional values and family structures are essential to a stable society"

## Labeling Process

### Step 1: Read Carefully

- Read the entire document, not just the headline
- Identify the main argument or thesis
- Note any explicit political positions taken

### Step 2: Identify Key Signals

**Policy Positions:**
- What policies are advocated for or against?
- What problems are identified and what solutions proposed?

**Framing:**
- How are issues described (e.g., "freedom" vs. "fairness")?
- What values are emphasized?
- What language is used (e.g., "tax relief" vs. "tax cuts for the wealthy")?

**Sources and Evidence:**
- Who is cited as authorities?
- What types of evidence are used?

### Step 3: Assign Label

Choose the label that best fits the **overall ideological stance** of the content.

**If uncertain:**
- Default to "Center/Mixed/Unclear" (class 1)
- Note your reasoning in comments (if collecting feedback)
- Flag for review by another labeler

### Step 4: Document Edge Cases

Keep notes on documents that:
- Are difficult to classify
- Mix ideological signals
- Represent new or unusual political positions
- Might reflect temporal shifts in political discourse

## Common Pitfalls to Avoid

### 1. Topic ≠ Ideology

**Wrong:** "This article discusses immigration, so it's right-wing"
**Right:** "This article advocates for open borders → left-wing"

Topics can be discussed from any ideological perspective.

### 2. Style ≠ Substance

**Wrong:** "This is written politely, so it's centrist"
**Right:** Focus on the actual policy positions, not tone

### 3. Source Bias

**Wrong:** "This is from a left-leaning outlet, so label it left"
**Right:** Read the content and label it based on what it says

Even partisan outlets publish diverse content.

### 4. Personal Agreement

**Wrong:** "I disagree with this, so it's [opposite of my views]"
**Right:** Label objectively based on the guidelines

Your personal beliefs should not influence labeling.

### 5. Complexity Avoidance

**Wrong:** "This is complicated, I'll just pick left or right"
**Right:** If genuinely mixed/unclear, use the center label

The center category is important and valid.

## Quality Control

### Inter-rater Reliability

- Multiple labelers should label a subset of documents
- Calculate agreement (Cohen's kappa or similar)
- Discuss disagreements to refine guidelines
- Target: >70% agreement, >0.6 kappa

### Calibration Sessions

- Regular meetings to discuss difficult cases
- Review borderline examples as a team
- Update guidelines based on learnings

### Documentation

For each labeling session, record:
- Number of documents labeled
- Time taken
- Difficult cases encountered
- Questions or guideline ambiguities

## Example Annotation Format

CSV format: `doc_id,label,confidence,notes`

```csv
doc_id,label,confidence,notes
doc_001,0,high,Clear progressive policy advocacy
doc_002,1,medium,Balanced presentation of healthcare debate
doc_003,2,high,Explicit conservative positions
doc_004,1,low,Mixed signals - flag for review
```

Confidence levels:
- `high`: Clear and unambiguous
- `medium`: Some uncertainty but label seems correct
- `low`: Genuinely unsure, needs review

## Testing Your Understanding

Before labeling production data, practice with these examples:

**Example 1:**
"The government should invest heavily in renewable energy to combat climate change and create green jobs."

**Label:** 0 (Left) - Advocates for government action on climate

**Example 2:**
"A new poll shows 52% of Americans support the policy, while 48% oppose it. Experts disagree on the economic impact."

**Label:** 1 (Center) - Factual reporting without advocacy

**Example 3:**
"High taxes and excessive regulation stifle business innovation and harm economic growth."

**Label:** 2 (Right) - Free market, anti-regulation position

## Updates and Feedback

These guidelines are living documents. If you encounter:
- Systematic ambiguities
- New political phenomena not covered
- Conflicting guidance

Please report to [your contact] for guideline updates.

## Final Reminder

You are labeling **text content**, not people. This is about discourse analysis, not individual profiling. Labels must never be used to:
- Infer individual beliefs
- Target individuals
- Make decisions affecting rights

Thank you for your careful and ethical work!
