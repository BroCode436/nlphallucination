"""
run_pipeline.py
Unified Command-Line Pipeline Entry Point.

Usage:
    py -3.12 run_pipeline.py --all           # Runs tests, checks data/checkpoints, and evaluates
    py -3.12 run_pipeline.py --test          # Runs automated regression test suite
    py -3.12 run_pipeline.py --preprocess    # Runs data preprocessing with leakage assertions
    py -3.12 run_pipeline.py --train         # Runs baseline and transformer detector training
    py -3.12 run_pipeline.py --evaluate      # Runs comprehensive benchmarking and ablations
"""

import argparse
import sys
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'checkpoints')
sys.path.insert(0, SRC_DIR)


def run_unit_tests():
    print("\n[STEP 1/3] Running Automated Unit & Regression Tests...")
    cmd = [sys.executable, "-m", "unittest", "discover", "tests"]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("[ERROR] Test suite failed!")
        sys.exit(1)
    print("[SUCCESS] All tests passed!")


def run_preprocessing():
    print("\n[STEP] Running Data Preprocessing and Leak-Free Splitting...")
    from preprocessing import run_pipeline as prep_pipeline
    prep_pipeline()


def run_training():
    print("\n[STEP] Running Baseline and DistilBERT Detector Training...")
    from train import run_training_pipeline
    run_training_pipeline()


def run_evaluation():
    print("\n[STEP] Running Research-Grade Model Evaluation, Benchmarking & Ablations...")
    from evaluate import run_full_evaluation
    run_full_evaluation()


def check_prerequisites_exist():
    """Checks if processed data and checkpoints are already available."""
    req_data = [
        os.path.join(PROCESSED_DIR, 'ragtruth_claims_train.csv'),
        os.path.join(PROCESSED_DIR, 'ragtruth_claims_test.csv')
    ]
    data_ready = all(os.path.exists(p) for p in req_data)

    req_checkpoints = [
        os.path.join(CHECKPOINTS_DIR, 'best_baseline_clf.joblib'),
        os.path.join(CHECKPOINTS_DIR, 'best_baseline_vec.joblib'),
        os.path.join(CHECKPOINTS_DIR, 'best_transformer_detector')
    ]
    models_ready = all(os.path.exists(p) for p in req_checkpoints)

    return data_ready, models_ready


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Evidence-Guided Hallucination Detection & Correction Pipeline"
    )
    parser.add_argument('--all', action='store_true',
                        help="Run test suite, verify data/checkpoints (train only if missing), and run full evaluation")
    parser.add_argument('--test', action='store_true',
                        help="Run automated unit and regression test suite")
    parser.add_argument('--preprocess', action='store_true',
                        help="Run data preprocessing and leak-free split generation")
    parser.add_argument('--train', action='store_true',
                        help="Explicitly train baseline and transformer detectors")
    parser.add_argument('--force-train', action='store_true',
                        help="Force model retraining during --all even if checkpoints exist")
    parser.add_argument('--evaluate', action='store_true',
                        help="Run evaluation suite, benchmarks, and ablation studies")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.all:
        run_unit_tests()
        data_ready, models_ready = check_prerequisites_exist()

        if not data_ready:
            print("[INFO] Processed dataset missing. Running preprocessing...")
            run_preprocessing()
        else:
            print("[INFO] Processed dataset found. Skipping redundant preprocessing.")

        if args.force_train or not models_ready:
            print("[INFO] Model checkpoints missing or --force-train specified. Running training...")
            run_training()
        else:
            print("[INFO] Saved model checkpoints found. Skipping redundant retraining.")

        run_evaluation()

    elif args.test:
        run_unit_tests()
    elif args.preprocess:
        run_preprocessing()
    elif args.train:
        run_training()
    elif args.evaluate:
        data_ready, _ = check_prerequisites_exist()
        if not data_ready:
            run_preprocessing()
        run_evaluation()


if __name__ == "__main__":
    main()
