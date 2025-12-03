#!/usr/bin/env python3
"""
Visualization script for experiment results.

Usage:
    python visualize_results.py experiment_results.json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load experiment results from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_rq1_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Plot RQ1 results: Federated vs Isolated"""
    if "RQ1" not in results:
        return
    
    rq1_data = results["RQ1"]
    federated = rq1_data.get("federated_results", [])
    isolated = rq1_data.get("isolated_results", [])
    
    if not federated or not isolated:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RQ1: Federated LoRA vs Isolated Local Fine-tuning", fontsize=16, fontweight='bold')
    
    # Extract metrics
    fed_rounds = [r["round_num"] for r in federated]
    fed_quality = [r["avg_code_quality"] for r in federated]
    fed_exact_match = [r["avg_exact_match"] for r in federated]
    fed_exec_pass = [r["avg_execution_pass_rate"] for r in federated]
    fed_comm = [r["total_communication_bytes"] for r in federated]
    
    iso_rounds = [r["round_num"] for r in isolated]
    iso_quality = [r["avg_code_quality"] for r in isolated]
    iso_exact_match = [r["avg_exact_match"] for r in isolated]
    iso_exec_pass = [r["avg_execution_pass_rate"] for r in isolated]
    iso_comm = [r["total_communication_bytes"] for r in isolated]
    
    # Plot 1: Code Quality
    ax = axes[0, 0]
    ax.plot(fed_rounds, fed_quality, 'o-', label='Federated', linewidth=2, markersize=6)
    ax.plot(iso_rounds, iso_quality, 's-', label='Isolated', linewidth=2, markersize=6)
    ax.set_xlabel('Round')
    ax.set_ylabel('Average Code Quality')
    ax.set_title('Code Generation Quality Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Exact Match Rate
    ax = axes[0, 1]
    ax.plot(fed_rounds, fed_exact_match, 'o-', label='Federated', linewidth=2, markersize=6)
    ax.plot(iso_rounds, iso_exact_match, 's-', label='Isolated', linewidth=2, markersize=6)
    ax.set_xlabel('Round')
    ax.set_ylabel('Exact Match Rate')
    ax.set_title('Exact Match Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Execution Pass Rate
    ax = axes[1, 0]
    ax.plot(fed_rounds, fed_exec_pass, 'o-', label='Federated', linewidth=2, markersize=6)
    ax.plot(iso_rounds, iso_exec_pass, 's-', label='Isolated', linewidth=2, markersize=6)
    ax.set_xlabel('Round')
    ax.set_ylabel('Execution Pass Rate')
    ax.set_title('Code Execution Success Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Communication Cost
    ax = axes[1, 1]
    ax.plot(fed_rounds, fed_comm, 'o-', label='Federated', linewidth=2, markersize=6)
    ax.plot(iso_rounds, iso_comm, 's-', label='Isolated', linewidth=2, markersize=6)
    ax.set_xlabel('Round')
    ax.set_ylabel('Communication (bytes)')
    ax.set_title('Communication Cost Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "rq1_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Summary statistics
    print("\nRQ1 Summary Statistics:")
    print(f"  Federated - Avg Quality: {np.mean(fed_quality):.4f}, Final: {fed_quality[-1]:.4f}")
    print(f"  Isolated  - Avg Quality: {np.mean(iso_quality):.4f}, Final: {iso_quality[-1]:.4f}")
    improvement = (np.mean(fed_quality) - np.mean(iso_quality)) / (np.mean(iso_quality) + 1e-6) * 100
    print(f"  Improvement: {improvement:.2f}%")


def plot_rq2_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Plot RQ2 results: Communication vs Accuracy Trade-off"""
    if "RQ2" not in results:
        return
    
    rq2_data = results["RQ2"]
    analysis = rq2_data.get("rq2_analysis", {})
    configs = analysis.get("configurations", [])
    pareto = analysis.get("pareto_frontier", [])
    
    if not configs:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RQ2: Communication Cost vs Accuracy Trade-off", fontsize=16, fontweight='bold')
    
    # Extract data
    config_names = [c["config"] for c in configs]
    qualities = [c["avg_quality"] for c in configs]
    comms = [c["avg_communication_bytes"] for c in configs]
    
    pareto_names = [c["config"] for c in pareto]
    pareto_qualities = [c["avg_quality"] for c in pareto]
    pareto_comms = [c["avg_communication_bytes"] for c in pareto]
    
    # Plot 1: All configurations
    ax = axes[0]
    ax.scatter(comms, qualities, s=100, alpha=0.6, label='All configs')
    if pareto:
        ax.scatter(pareto_comms, pareto_qualities, s=200, marker='*', 
                  color='red', label='Pareto frontier', zorder=5)
        # Draw Pareto frontier
        sorted_indices = np.argsort(pareto_comms)
        ax.plot(np.array(pareto_comms)[sorted_indices], 
               np.array(pareto_qualities)[sorted_indices],
               'r--', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Communication (bytes)')
    ax.set_ylabel('Average Quality')
    ax.set_title('Communication vs Quality Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Quality per byte
    ax = axes[1]
    quality_per_byte = [c["quality_per_byte"] for c in configs]
    colors = ['red' if c["config"] in pareto_names else 'blue' for c in configs]
    ax.bar(range(len(configs)), quality_per_byte, color=colors, alpha=0.6)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Quality per Byte')
    ax.set_title('Efficiency Metric')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([c.split('_')[1] for c in config_names], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / "rq2_tradeoff.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Summary
    print("\nRQ2 Summary Statistics:")
    print(f"  Total configurations tested: {len(configs)}")
    print(f"  Pareto frontier size: {len(pareto)}")
    if pareto:
        print("  Pareto frontier configurations:")
        for c in pareto:
            print(f"    {c['config']}: Quality={c['avg_quality']:.4f}, "
                  f"Comm={c['avg_communication_bytes']:.0f} bytes")


def plot_rq3_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Plot RQ3 results: Hierarchical Aggregation"""
    if "RQ3" not in results:
        return
    
    rq3_data = results["RQ3"]
    flat_results = rq3_data.get("flat_results", [])
    hier_results = rq3_data.get("hierarchical_results", [])
    skill_results = rq3_data.get("skill_aware_results", [])
    
    if not flat_results or not hier_results or not skill_results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RQ3: Hierarchical Aggregation for Specialization", fontsize=16, fontweight='bold')
    
    # Extract metrics
    flat_rounds = [r["round_num"] for r in flat_results]
    flat_quality = [r["avg_code_quality"] for r in flat_results]
    flat_spec = [r.get("avg_code_quality", 0) for r in flat_results]  # Simplified
    
    hier_rounds = [r["round_num"] for r in hier_results]
    hier_quality = [r["avg_code_quality"] for r in hier_results]
    hier_spec = [r.get("avg_code_quality", 0) for r in hier_results]
    
    skill_rounds = [r["round_num"] for r in skill_results]
    skill_quality = [r["avg_code_quality"] for r in skill_results]
    skill_spec = [r.get("avg_code_quality", 0) for r in skill_results]
    
    # Plot 1: Quality comparison
    ax = axes[0, 0]
    ax.plot(flat_rounds, flat_quality, 'o-', label='Flat', linewidth=2, markersize=6)
    ax.plot(hier_rounds, hier_quality, 's-', label='Hierarchical', linewidth=2, markersize=6)
    ax.plot(skill_rounds, skill_quality, '^-', label='Skill-Aware', linewidth=2, markersize=6)
    ax.set_xlabel('Round')
    ax.set_ylabel('Average Code Quality')
    ax.set_title('Quality Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Specialization tracking
    ax = axes[0, 1]
    ax.plot(flat_rounds, flat_spec, 'o-', label='Flat', linewidth=2, markersize=6)
    ax.plot(hier_rounds, hier_spec, 's-', label='Hierarchical', linewidth=2, markersize=6)
    ax.plot(skill_rounds, skill_spec, '^-', label='Skill-Aware', linewidth=2, markersize=6)
    ax.set_xlabel('Round')
    ax.set_ylabel('Specialization Score')
    ax.set_title('Specialization Preservation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final metrics comparison
    ax = axes[1, 0]
    strategies = ['Flat', 'Hierarchical', 'Skill-Aware']
    final_qualities = [flat_quality[-1] if flat_quality else 0,
                      hier_quality[-1] if hier_quality else 0,
                      skill_quality[-1] if skill_quality else 0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar(strategies, final_qualities, color=colors, alpha=0.7)
    ax.set_ylabel('Final Code Quality')
    ax.set_title('Final Quality Comparison')
    ax.set_ylim([0, max(final_qualities) * 1.2])
    for bar, val in zip(bars, final_qualities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Comparison metrics
    ax = axes[1, 1]
    comparison = rq3_data.get("rq3_comparison", {})
    metrics_names = ['avg_quality', 'specialization_preservation']
    x = np.arange(len(strategies))
    width = 0.35
    
    for i, metric in enumerate(metrics_names):
        values = [comparison.get(s.lower().replace('-', '_'), {}).get(metric, 0) 
                 for s in strategies]
        ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
    
    ax.set_ylabel('Score')
    ax.set_title('Aggregated Metrics')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(strategies)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / "rq3_hierarchical.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Summary
    print("\nRQ3 Summary Statistics:")
    for strategy, metrics in comparison.items():
        print(f"  {strategy}:")
        for key, value in metrics.items():
            print(f"    {key}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("results_file", help="Path to experiment_results.json")
    parser.add_argument("--output-dir", type=str, default="plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return 1
    
    # Load results
    results = load_results(args.results_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading results from: {args.results_file}")
    print(f"Output directory: {output_dir}")
    
    # Generate plots
    plot_rq1_results(results, output_dir)
    plot_rq2_results(results, output_dir)
    plot_rq3_results(results, output_dir)
    
    print("\nVisualization complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

