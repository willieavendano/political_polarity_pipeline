# Model Card: Political Polarity Classifier

## Model Details

**Model Type:** Text Classification (3-class)

**Architecture Options:**
- Baseline: TF-IDF features + Calibrated Logistic Regression
- Transformer: Fine-tuned DistilBERT-base-uncased

**Version:** 1.0

**Date:** January 2026

**Developers:** [Your Organization/Name]

## Intended Use

### Primary Use Cases (AGGREGATE ANALYSIS ONLY)

✅ **Permitted:**
- Analyzing political discourse patterns in news corpora at scale
- Comparing aggregate media coverage across outlets or regions
- Research on political communication trends over time
- Content categorization for media studies
- Bias detection in large document collections

### Prohibited Uses

❌ **Forbidden:**
- Inferring or predicting individual political beliefs or affiliations
- Making decisions affecting individual rights, employment, or opportunities
- Targeting individuals with political advertising or content
- Profiling individuals based on their writing
- Inferring protected characteristics (race, religion, ethnicity, etc.)
- Any use that violates privacy, discrimination, or election laws

## Training Data

### Sources
- **Distant Supervision:** Outlet-level bias labels (weak supervision)
  - Assumes outlet stance approximates article stance (noisy but scalable)
- **Manual Labels:** Optional human-curated document labels
- **Size:** Varies by deployment (recommend 10k+ documents for production)

### Label Scheme
- **Class 0:** Left/Democrat/Liberal
- **Class 1:** Center/Mixed/Unclear  
- **Class 2:** Right/Republican/Conservative

**Important:** The "Center" class captures:
- Genuinely centrist/balanced content
- Mixed ideological signals
- Unclear or ambiguous stance
- Non-political content misclassified during filtering

### Data Limitations
- English language only
- Primarily U.S. political context
- Historical data reflects past political discourse (may not generalize to future)
- Outlet labels introduce systematic noise
- Class imbalance common (depends on data sources)

## Evaluation

### Metrics (Example - Actual Performance Varies)

**Baseline Model:**
- Accuracy: ~70-80% (depends on data quality)
- Macro F1: ~0.65-0.75
- Strengths: Fast, interpretable, good calibration
- Weaknesses: Ignores context, sensitive to topic shifts

**Transformer Model:**
- Accuracy: ~75-85%
- Macro F1: ~0.70-0.80
- Strengths: Captures context, better on subtle cases
- Weaknesses: Slower, more data-hungry, less interpretable

### Evaluation Protocol
- Stratified train/val/test split (70/15/15)
- Stratification by source to reduce leakage
- Calibration assessed via Expected Calibration Error (ECE)
- Temporal validation recommended (train on past, test on recent)

## Limitations and Biases

### Known Limitations

1. **Topic Conflation Risk**
   - Model may learn topic associations rather than ideological stance
   - Example: Climate change mentioned ≠ left-leaning stance
   - Mitigation: Include diverse topics in all classes

2. **Outlet Label Noise**
   - Outlet bias ≠ article bias
   - Op-eds, guest columns, reporting styles vary
   - Mitigation: Use manual labels where possible

3. **Temporal Drift**
   - Political language evolves rapidly
   - Models trained on old data may degrade
   - Mitigation: Regular retraining with recent data

4. **Style vs. Substance**
   - Model may learn writing style markers instead of ideological content
   - Example: Formal tone ≠ centrist stance
   - Mitigation: Diverse training sources per class

5. **Class Imbalance**
   - "Center" class often underrepresented
   - May lead to over-prediction of left/right
   - Mitigation: Class weighting, resampling, or threshold tuning

6. **Context Window Limits**
   - Transformers have token limits (256-512 tokens)
   - Long documents truncated, potentially losing key context
   - Mitigation: Document segmentation or hierarchical approaches

### Potential Biases

- **Source Bias:** Reflects ideological distribution of training outlets
- **Topic Bias:** Overrepresented topics may drive predictions
- **Recency Bias:** Recent political events may dominate learned patterns
- **Geographic Bias:** U.S.-centric; may not apply to other countries
- **Socioeconomic Bias:** Reflects media accessible to data collectors

## Ethical Considerations

### Privacy

- **K-Anonymity:** Subset reports enforce minimum group size (default: 30)
- **No Individual Inference:** System designed for aggregate analysis only
- **Metadata Restrictions:** Only uses externally provided metadata (never infers)

### Fairness

- **No Protected Attribute Inference:** Does not and must not infer race, religion, gender, etc.
- **Audit Recommendations:** Periodically check for disparate impact if used in high-stakes contexts
- **Transparency:** Phrase extraction provides interpretability into predictions

### Misuse Potential

- **Manipulation Risk:** Could be used to game recommendations or amplification
- **Echo Chamber Risk:** Automated filtering may increase polarization
- **Disinformation Risk:** Adversaries could exploit model weaknesses

**Mitigation:** Clear usage guidelines, access controls, and monitoring

## Uncertainty and Confidence

- All predictions include probability distributions
- Entropy metric quantifies uncertainty
- Calibration ensures probabilities reflect true likelihood
- **High uncertainty (H > 0.8):** Unclear or mixed content
- **Low confidence (<0.5 max prob):** Flag for review

## Reproducibility

- **Random Seed:** Fixed at 42 for all operations
- **Dependency Versions:** Pinned in requirements.txt
- **Data Splits:** Saved and reused across experiments
- **Model Checkpoints:** Full model state saved for inference

## Maintenance and Updates

### Retraining Triggers
- Performance degradation on held-out validation set
- Significant political events or language shifts
- New data sources or label corrections
- User feedback indicating systematic errors

### Monitoring
- Track prediction distributions over time
- Monitor calibration on recent data
- Review error analysis for emerging patterns
- Check for topic or source drift

## Contact and Feedback

For questions about:
- **Ethical use:** [ethics@example.com]
- **Technical issues:** [support@example.com]
- **Research collaboration:** [research@example.com]

## Version History

- **v1.0 (Jan 2026):** Initial release with baseline and transformer models

## References

- Preoţiuc-Pietro, D., et al. (2017). Beyond binary labels: Political ideology prediction of Twitter users.
- Baly, R., et al. (2020). What was written vs. who read it: News media profiling using text analysis and social media context.
- Ethical NLP guidelines: [ACL Ethics](https://www.aclweb.org/portal/content/acl-code-ethics)

## Disclaimer

This model is an analytical tool, not a source of truth. Political ideology is complex, multidimensional, and context-dependent. Model predictions should be interpreted as rough approximations for aggregate analysis, not definitive labels. Always combine model outputs with domain expertise and human judgment.
