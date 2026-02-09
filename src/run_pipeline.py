#!/usr/bin/env python3
"""
CLI Entry Point for FETCH Range Calculation Pipeline

Usage:
    python -m src.run_pipeline --config src/config_example.yaml
    python -m src.run_pipeline --stations 2502 2503 2504 --output-dir results/
    python -m src.run_pipeline --test
"""

import argparse
import sys
from pathlib import Path

from .config import PipelineConfig, load_config
from .pipeline import run_full_pipeline, run_quick_test, validate_inputs


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FETCH Range Calculation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.run_pipeline --config src/config_example.yaml
  python -m src.run_pipeline --output-dir results/
  python -m src.run_pipeline --test
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--stations",
        type=str,
        nargs="+",
        default=["2502", "2503", "2504"],
        help="Station identifiers to process",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/",
        help="Output directory for results",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test to verify inputs",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate input files exist without running pipeline",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )

    parser.add_argument(
        "--generate-config",
        type=str,
        metavar="PATH",
        help="Generate example config file at specified path",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Generate config file if requested
    if args.generate_config:
        config = PipelineConfig()
        config.to_yaml(args.generate_config)
        print(f"Generated config file: {args.generate_config}")
        return 0

    # Load configuration
    if args.config:
        try:
            config = load_config(args.config)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
    else:
        config = PipelineConfig()

    # Override output directory if specified
    if args.output_dir != "results/":
        config.output_dir = args.output_dir

    # Validate inputs
    if args.validate:
        missing = validate_inputs(config)
        if missing:
            print("Missing input files:")
            for path in missing:
                print(f"  - {path}")
            return 1
        else:
            print("All input files found.")
            return 0

    # Run quick test
    if args.test:
        success = run_quick_test(config)
        return 0 if success else 1

    # Run full pipeline
    try:
        results = run_full_pipeline(
            config=config,
            verbose=not args.quiet,
        )

        # Print summary
        if not args.quiet:
            print("\nResults summary:")
            if "distance_dfs" in results:
                print(f"  Distance files: {len(results['distance_dfs'])}")
            if "triangle_df" in results:
                print(f"  Triangle area: {len(results['triangle_df'])} points")
            if "figures" in results:
                print(f"  Figures generated: {len(results['figures'])}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
