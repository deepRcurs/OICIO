"""
OICIO CLI - Command Line Interface
Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh

Usage:
  python -m oicio.cli ingest --file long_doc.txt
  python -m oicio.cli query --question "How many entity?"
  python -m oicio.cli eval --benchmark oolong --samples 10
  python -m oicio.cli train --epochs 2
"""

import sys
sys.path.insert(0, '/home/user')
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="OICIO - Optimized Infinite Context Intelligence Orchestration")
    parser.add_argument("--version", action="store_true", help="Show version and credits")
    
    subparsers = parser.add_subparsers(dest="command")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest long document")
    ingest_parser.add_argument("--file", type=str, help="File to ingest")
    ingest_parser.add_argument("--tokens", type=int, default=1000, help="Synthetic tokens if no file")

    # query
    query_parser = subparsers.add_parser("query", help="Query OICIO")
    query_parser.add_argument("--question", type=str, required=True, help="Question")

    # eval
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--benchmark", type=str, default="oolong", choices=["oolong", "longbench"])
    eval_parser.add_argument("--samples", type=int, default=2)

    # train
    train_parser = subparsers.add_parser("train", help="Train ternary model")
    train_parser.add_argument("--epochs", type=int, default=2)

    # demo
    demo_parser = subparsers.add_parser("demo", help="Run full demo")

    args = parser.parse_args()

    if args.version:
        print("OICIO v0.1 POC")
        print("Credits: deepRcurs Labs @deeprcurs")
        print("Author: Mzed Imamkh @mzedimamkh")
        print("Paradigm: Frontier-quality at 1.58-bit with harness recursion")
        print("Snapshot: 200KB code, toolchain in .venv (excluded)")
        return

    if args.command == "ingest":
        from oicio.runtime.oicio_runtime import OICIORuntime
        runtime = OICIORuntime(dim=64)
        if args.file and os.path.exists(args.file):
            with open(args.file, 'r') as f:
                docs = [line.strip() for line in f if line.strip()]
        else:
            # synthetic
            docs = [f"user_{i}: entity data" if i%3==0 else f"log {i}: system" for i in range(args.tokens)]
        runtime.ingest_document(docs)
        print(f"Ingested {len(docs)} chunks")

    elif args.command == "query":
        from oicio.runtime.oicio_runtime import OICIORuntime
        runtime = OICIORuntime(dim=64)
        # Need to have ingested first, for POC generate synthetic
        docs = [f"user_{i}: entity data" if i%3==0 else f"log {i}: system" for i in range(1000)]
        runtime.ingest_document(docs)
        result = runtime.query(args.question)
        print(f"Answer: {result}")

    elif args.command == "eval":
        from oicio.eval.oolong_eval import OOLONGEval
        evaluator = OOLONGEval()
        evaluator.run_eval(num_samples_per_bucket=args.samples)

    elif args.command == "train":
        from oicio.training.qat_trainer import QATTrainer, SyntheticOOLONGDataset
        from oicio.core.ternary_san import TernarySAN
        model = TernarySAN(vocab_size=1000, dim=128, num_layers=2, num_heads=4)
        dataset = SyntheticOOLONGDataset(num_samples=200, seq_len=64)
        trainer = QATTrainer(model, dataset, lr=1e-3)
        trainer.train(epochs=args.epochs, batch_size=8)

    elif args.command == "demo":
        import subprocess
        subprocess.run([sys.executable, "/home/user/oicio/demo/oicio_full_demo.py"])

    else:
        parser.print_help()
        print("\nCredits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh")

if __name__ == "__main__":
    main()
