"""
Generate privacy-preserving aggregate reports by subset.

Implements k-anonymity to ensure no small groups are exposed.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.privacy import KAnonymityEnforcer, AggregateReporter, create_privacy_notice

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> list:
    """Load JSONL file."""
    documents = []
    with open(file_path, 'r') as f:
        for line in f:
            doc = json.loads(line.strip())
            documents.append(doc)
    return documents


def plot_subset_polarity(
    report_data: dict,
    output_path: str,
    class_names: list,
):
    """Plot polarity scores by subset."""
    subsets = report_data.get('subsets', {})
    
    if not subsets:
        logger.warning("No subsets to plot")
        return
    
    # Extract data
    subset_names = []
    mean_scores = []
    counts = []
    
    for subset_name, stats in subsets.items():
        subset_names.append(subset_name)
        mean_scores.append(stats['polarity_score']['mean'])
        counts.append(stats['count'])
    
    # Sort by polarity score
    sorted_indices = sorted(range(len(mean_scores)), key=lambda i: mean_scores[i])
    subset_names = [subset_names[i] for i in sorted_indices]
    mean_scores = [mean_scores[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(subset_names) * 0.4)))
    
    colors = ['#2E86AB' if score < -0.1 else '#A23B72' if score > 0.1 else '#808080' 
              for score in mean_scores]
    
    bars = ax.barh(range(len(subset_names)), mean_scores, color=colors)
    ax.set_yticks(range(len(subset_names)))
    ax.set_yticklabels(subset_names)
    ax.set_xlabel('Mean Polarity Score (Left ← → Right)')
    ax.set_title(f'Political Polarity by {report_data["subset_key"]}')
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        label_x = width + 0.01 if width > 0 else width - 0.01
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x, i, f'n={count}', va='center', ha=ha, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved polarity plot to {output_path}")


def plot_subset_distribution(
    report_data: dict,
    output_path: str,
    class_names: list,
):
    """Plot class distribution by subset."""
    subsets = report_data.get('subsets', {})
    
    if not subsets:
        logger.warning("No subsets to plot")
        return
    
    # Extract data
    subset_names = list(subsets.keys())
    class_proportions = {class_name: [] for class_name in class_names}
    
    for subset_name in subset_names:
        stats = subsets[subset_name]
        for class_name in class_names:
            prop = stats['class_distribution'][class_name]['proportion']
            class_proportions[class_name].append(prop)
    
    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(10, max(6, len(subset_names) * 0.4)))
    
    colors = ['#2E86AB', '#808080', '#A23B72']  # Blue, Gray, Red
    left = [0] * len(subset_names)
    
    for i, class_name in enumerate(class_names):
        props = class_proportions[class_name]
        ax.barh(range(len(subset_names)), props, left=left, 
                label=class_name, color=colors[i])
        left = [l + p for l, p in zip(left, props)]
    
    ax.set_yticks(range(len(subset_names)))
    ax.set_yticklabels(subset_names)
    ax.set_xlabel('Proportion')
    ax.set_title(f'Class Distribution by {report_data["subset_key"]}')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved distribution plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate privacy-preserving subset reports"
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Path to predictions JSONL (with subset metadata)"
    )
    parser.add_argument(
        "--subset_key",
        type=str,
        required=True,
        help="Metadata key to group by (e.g., 'region', 'age_bucket')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for reports"
    )
    parser.add_argument(
        "--min_group_size",
        type=int,
        default=30,
        help="Minimum group size for k-anonymity"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Generating Privacy-Preserving Subset Report")
    logger.info("=" * 60)
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    label_config = config.get('labels', {})
    class_names = label_config.get('class_names', [
        "Left/Democrat/Liberal",
        "Center/Mixed/Unclear",
        "Right/Republican/Conservative",
    ])
    
    privacy_config = config.get('privacy', {})
    if args.min_group_size is None:
        args.min_group_size = privacy_config.get('min_group_size', 30)
    
    # Step 1: Load predictions
    logger.info(f"\n[1/4] Loading predictions from {args.predictions_path}...")
    documents = load_jsonl(args.predictions_path)
    logger.info(f"Loaded {len(documents)} documents")
    
    # Convert to DataFrame
    data_records = []
    for doc in documents:
        if doc.get('predicted_class') is None:
            continue
        
        # Extract subset metadata
        subset_meta = doc.get('subset_meta', {})
        if args.subset_key not in subset_meta:
            continue
        
        record = {
            'doc_id': doc.get('doc_id', ''),
            'predicted_class': doc['predicted_class'],
            'polarity_score': doc.get('polarity_score', 0),
            'uncertainty': doc.get('uncertainty', 0),
            'keywords': doc.get('keywords', []),
            args.subset_key: subset_meta[args.subset_key],
        }
        data_records.append(record)
    
    if not data_records:
        logger.error(f"No documents with subset_meta['{args.subset_key}'] found!")
        return 1
    
    df = pd.DataFrame(data_records)
    logger.info(f"Valid documents with '{args.subset_key}': {len(df)}")
    
    # Step 2: Generate report with k-anonymity
    logger.info(f"\n[2/4] Generating report with k-anonymity (k={args.min_group_size})...")
    
    k_anonymity = KAnonymityEnforcer(min_group_size=args.min_group_size)
    reporter = AggregateReporter(k_anonymity)
    
    polarity_report = reporter.generate_polarity_report(
        df, args.subset_key, class_names
    )
    
    phrase_report = reporter.generate_phrase_report(
        df, args.subset_key, top_n=10
    )
    
    # Step 3: Generate visualizations
    logger.info("\n[3/4] Generating visualizations...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    if polarity_report.get('subsets'):
        # Polarity plot
        polarity_plot_path = os.path.join(
            args.output_dir, f"polarity_by_{args.subset_key}.png"
        )
        plot_subset_polarity(polarity_report, polarity_plot_path, class_names)
        
        # Distribution plot
        dist_plot_path = os.path.join(
            args.output_dir, f"distribution_by_{args.subset_key}.png"
        )
        plot_subset_distribution(polarity_report, dist_plot_path, class_names)
    
    # Step 4: Save reports
    logger.info("\n[4/4] Saving reports...")
    
    # Polarity report
    polarity_path = os.path.join(args.output_dir, f"polarity_report_{args.subset_key}.json")
    with open(polarity_path, 'w') as f:
        json.dump(polarity_report, f, indent=2)
    logger.info(f"Saved polarity report to {polarity_path}")
    
    # Phrase report
    phrase_path = os.path.join(args.output_dir, f"phrase_report_{args.subset_key}.json")
    with open(phrase_path, 'w') as f:
        json.dump(phrase_report, f, indent=2)
    logger.info(f"Saved phrase report to {phrase_path}")
    
    # Privacy notice
    notice = create_privacy_notice()
    notice_path = os.path.join(args.output_dir, "PRIVACY_NOTICE.txt")
    with open(notice_path, 'w') as f:
        f.write(notice)
    logger.info(f"Saved privacy notice to {notice_path}")
    
    # Summary statistics
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("SUBSET ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Subset Key: {args.subset_key}\n")
        f.write(f"Total Documents: {len(df)}\n")
        f.write(f"K-Anonymity Threshold: {args.min_group_size}\n")
        f.write(f"Valid Subsets: {polarity_report.get('total_subsets', 0)}\n")
        f.write(f"Suppressed Groups: {len(polarity_report.get('suppressed_groups', []))}\n\n")
        
        if polarity_report.get('suppressed_groups'):
            f.write(f"Suppressed (< {args.min_group_size} samples):\n")
            for group in polarity_report['suppressed_groups']:
                f.write(f"  - {group}\n")
            f.write("\n")
        
        f.write("Polarity by Subset:\n")
        f.write("-" * 60 + "\n")
        for subset_name, stats in polarity_report.get('subsets', {}).items():
            mean_score = stats['polarity_score']['mean']
            count = stats['count']
            f.write(f"\n{subset_name} (n={count}):\n")
            f.write(f"  Mean Polarity: {mean_score:.3f}\n")
            f.write(f"  Distribution:\n")
            for class_name, class_stats in stats['class_distribution'].items():
                count_cls = class_stats['count']
                prop = class_stats['proportion']
                f.write(f"    {class_name}: {count_cls} ({prop*100:.1f}%)\n")
    
    logger.info(f"Saved summary to {summary_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Subset report generation complete!")
    logger.info(f"Reports saved to {args.output_dir}")
    logger.info("\n⚠️  REMINDER: This is aggregate-level analysis only.")
    logger.info("Do NOT use for individual profiling or targeting.")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
