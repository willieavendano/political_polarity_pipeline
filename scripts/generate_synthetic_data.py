"""
Generate synthetic test data for the pipeline.

WARNING: This data is SYNTHETIC and for testing only.
"""

import json
import logging
import os
import random
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed
random.seed(42)

# Synthetic content templates
LEFT_TEMPLATES = [
    "Progressive policies are essential for economic equality. We need stronger labor protections and higher minimum wage.",
    "Climate change is an existential threat requiring immediate action. Green energy investments create jobs.",
    "Universal healthcare is a human right. Medicare for All would save lives and reduce costs.",
    "We must protect voting rights and expand access to the ballot box for all citizens.",
    "Income inequality has reached crisis levels. Tax reform should ask the wealthy to pay their fair share.",
]

CENTER_TEMPLATES = [
    "Both parties have valid points on this issue. We need bipartisan compromise to move forward.",
    "The data shows mixed results. More research is needed before drawing conclusions.",
    "There are tradeoffs to consider. Balance is key to effective policy.",
    "Experts disagree on the best approach. Multiple perspectives should be considered.",
    "This is a complex issue without easy answers. Pragmatic solutions require input from all sides.",
]

RIGHT_TEMPLATES = [
    "Free market principles and limited government intervention foster economic growth and prosperity.",
    "Second Amendment rights must be protected. Law-abiding citizens have the right to bear arms.",
    "Lower taxes and reduced regulations unleash entrepreneurship and create jobs.",
    "National security and strong borders are essential for protecting American sovereignty.",
    "School choice and parental rights empower families to make the best educational decisions.",
]

OUTLETS = {
    "ProgressiveDaily": "left",
    "LeftVoice": "left",
    "TheDemocrat": "left",
    "CenterNews": "center",
    "BalancedView": "center",
    "NeutralReport": "center",
    "ConservativePost": "right",
    "RightPerspective": "right",
    "TheRepublican": "right",
}

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
AGE_BUCKETS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]


def generate_document(doc_id: int, outlet: str, bias: str) -> dict:
    """Generate a synthetic document."""
    
    # Select template based on bias
    if bias == "left":
        text = random.choice(LEFT_TEMPLATES)
    elif bias == "center":
        text = random.choice(CENTER_TEMPLATES)
    else:
        text = random.choice(RIGHT_TEMPLATES)
    
    # Add some variation
    suffixes = [
        " This reflects the values of our community.",
        " Polls show strong support for this position.",
        " Recent events highlight the importance of this issue.",
        " Voters deserve to know where candidates stand.",
        " This policy would impact millions of Americans.",
    ]
    text += random.choice(suffixes)
    
    # Generate metadata
    date = datetime.now() - timedelta(days=random.randint(0, 365))
    
    doc = {
        "doc_id": f"doc_{doc_id:05d}",
        "text": text,
        "title": f"Opinion: {text.split('.')[0]}",
        "date": date.strftime("%Y-%m-%d"),
        "source": outlet,
        "url": f"https://{outlet.lower()}.com/article/{doc_id}",
        "subset_meta": {
            "region": random.choice(REGIONS),
            "age_bucket": random.choice(AGE_BUCKETS),
        }
    }
    
    return doc


def generate_corpus(num_docs: int = 300) -> list:
    """Generate synthetic corpus."""
    logger.info(f"Generating {num_docs} synthetic documents...")
    
    corpus = []
    outlets = list(OUTLETS.keys())
    
    for i in range(num_docs):
        outlet = random.choice(outlets)
        bias = OUTLETS[outlet]
        doc = generate_document(i, outlet, bias)
        corpus.append(doc)
    
    # Log distribution
    bias_counts = {}
    for doc in corpus:
        bias = OUTLETS[doc["source"]]
        bias_counts[bias] = bias_counts.get(bias, 0) + 1
    
    logger.info("Distribution:")
    for bias, count in sorted(bias_counts.items()):
        logger.info(f"  {bias}: {count} documents")
    
    return corpus


def generate_outlet_labels() -> str:
    """Generate outlet labels CSV content."""
    lines = ["outlet_name,bias_label"]
    for outlet, bias in OUTLETS.items():
        lines.append(f"{outlet},{bias}")
    return "\n".join(lines)


def main():
    """Generate synthetic data."""
    logger.info("=" * 60)
    logger.info("Generating Synthetic Test Data")
    logger.info("WARNING: This data is SYNTHETIC for testing only!")
    logger.info("=" * 60)
    
    # Create directories
    os.makedirs("./data/raw", exist_ok=True)
    
    # Generate corpus
    corpus = generate_corpus(num_docs=300)
    
    # Save corpus as JSONL
    corpus_path = "./data/raw/synthetic_corpus.jsonl"
    with open(corpus_path, 'w') as f:
        for doc in corpus:
            f.write(json.dumps(doc) + "\n")
    
    logger.info(f"Saved corpus to {corpus_path}")
    
    # Save outlet labels
    labels_path = "./data/raw/outlet_labels.csv"
    with open(labels_path, 'w') as f:
        f.write(generate_outlet_labels())
    
    logger.info(f"Saved outlet labels to {labels_path}")
    
    logger.info("=" * 60)
    logger.info("✓ Synthetic data generation complete!")
    logger.info("Next step: python -m scripts.prepare_data")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
