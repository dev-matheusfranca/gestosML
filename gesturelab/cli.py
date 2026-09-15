"""Interface de linha de comando para o núcleo, sem necessidade da UI."""
from __future__ import annotations

import argparse
import json
import signal
import threading
from pathlib import Path

from .ml import compare, evaluate_final, split_dataset, train
from .store import Store


def _store(args: argparse.Namespace) -> Store:
    return Store(Path(args.data_dir))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gesturelab", description="Laboratório local de gestos")
    parser.add_argument("--data-dir", default="data", help="Diretório local de dados (padrão: data)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="Valida o dataset atual")
    commands.add_parser("snapshot", help="Cria uma versão imutável do dataset")
    commands.add_parser("datasets", help="Lista versões e hashes")
    commands.add_parser("experiments", help="Lista experimentos e resultados")
    gui = commands.add_parser("gui", help="Abre a interface com o diretório de dados escolhido")
    gui.set_defaults(command="gui")
    split = commands.add_parser("split", help="Reserva divisão e lista IDs por partição; use o mesmo protocolo do treino")
    split.add_argument("dataset_id")
    split.add_argument("--protocol", choices=["session", "participant"], default="session")
    split.add_argument("--seed", type=int, default=42)
    export = commands.add_parser("export", help="Exporta o dataset atual para .parquet ou .csv")
    export.add_argument("path")
    train_parser = commands.add_parser("train", help="Treina modelos sem abrir a interface")
    train_parser.add_argument("dataset_id")
    train_parser.add_argument("--strategy", choices=["raw", "normalized", "distances", "angles"], default="normalized")
    train_parser.add_argument("--models", nargs="+", choices=["dummy", "logistic", "forest", "mlp"], default=None)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--protocol", choices=["session", "participant"], default="session")
    train_parser.add_argument("--preset", choices=["single", "quick"], default="quick")
    evaluate = commands.add_parser("evaluate", help="Avalia uma única vez o teste final reservado")
    evaluate.add_argument("experiment_id")
    comparison = commands.add_parser("compare", help="Compara experimentos e indica compatibilidade")
    comparison.add_argument("experiment_ids", nargs="+")
    args = parser.parse_args(argv)
    exit_code = 0
    try:
        if args.command == "gui":
            from .ui import run
            return run(args.data_dir)
        store = _store(args)
        if args.command == "validate":
            result = store.validate()
            exit_code = 0 if result["valid"] else 2
        elif args.command == "snapshot":
            result = {"dataset_id": store.snapshot()}
        elif args.command == "datasets":
            result = store.snapshots()
        elif args.command == "experiments":
            result = store.experiments()
        elif args.command == "split":
            frame, partitions = split_dataset(store, args.dataset_id, args.protocol, args.seed)
            result = {"dataset_id": args.dataset_id, "protocol": args.protocol,
                "partitions": {part: frame.loc[partitions == part, ["id", "session_id", "label", "participant"]].to_dict("records")
                    for part in ("train", "validation", "test")}}
        elif args.command == "export":
            result = {"path": str(store.export(args.path))}
        elif args.command == "train":
            cancel = threading.Event()
            previous_handler = signal.signal(signal.SIGINT, lambda signum, frame: cancel.set())
            try:
                ids = train(store, args.dataset_id, args.strategy, args.models, args.seed, args.protocol,
                            preset=args.preset, cancel=cancel, progress=print)
            finally:
                signal.signal(signal.SIGINT, previous_handler)
            result = {"experiments": ids, "cancelled": cancel.is_set()}
            exit_code = 130 if cancel.is_set() else (0 if len(ids) == len(args.models or ["dummy", "logistic", "forest", "mlp"]) else 2)
        elif args.command == "evaluate":
            result = evaluate_final(store, args.experiment_id)
        else:
            result = compare(store, args.experiment_ids)
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
