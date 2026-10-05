"""Explicit real-model smoke test. Default is a read-only local prerequisite report."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import types

package = types.ModuleType("omni_qualification_owned")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
from omni_qualification_owned import assets_store, logic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-text", default="")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--results-directory", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        inventory = assets_store.inventory(args.directory)
        inventory["inference_python_exists"] = args.python.is_file()
        inventory["reference_exists"] = args.reference is None or args.reference.is_file()
        inventory["environment_and_model_execution_checked"] = False
        if not args.execute:
            print(json.dumps(inventory, indent=2))
            return 0 if inventory["ready"] and inventory["inference_python_exists"] and inventory["reference_exists"] else 1
        if not inventory["ready"]: raise logic.OmniVoiceError("assets_missing")
        if not inventory["inference_python_exists"]: raise logic.OmniVoiceError("dependency")
        if not inventory["reference_exists"]: raise logic.OmniVoiceError("reference")
        if args.results_directory is None: parser.error("--execute requires --results-directory")
        results = args.results_directory.absolute(); results.mkdir(parents=True, exist_ok=True)
        if args.reference and not args.reference_text.strip(): parser.error("--reference requires --reference-text")
        cfg = logic.configuration({"voice_mode": "clone" if args.reference else "auto", "reference_text": args.reference_text,"asset_directory": str(args.directory.absolute()),
            "environment_python": str(args.python.absolute()), "device": args.device, "precision": args.precision,
            "reference_audio": str(args.reference.absolute()) if args.reference else "", "output_directory": str(results / "audio")})
        records = []
        for language, text in (("en", "This is a local speech synthesis test. The report is ready."),
                               ("fr", "Ceci est un test de synthèse vocale locale. Le rapport est prêt.")):
            with TemporaryDirectory(prefix="omni-qualification-", dir=results) as temporary:
                prepared = logic.generate({"text": text, "request_id": "real-model-" + language.lower()},
                    {**cfg, "language": language}, results, Path(temporary))
                try: metadata = logic.publish(prepared)
                finally: prepared.close()
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", metadata["path"]]))
            stream = probe["streams"][0]
            if stream["codec_name"] != "opus" or int(stream["sample_rate"]) != 48000 or stream["channels"] != 1:
                raise logic.OmniVoiceError("audio_invalid")
            records.append(metadata)
        print(json.dumps({"status": "passed", "scope": "actual model smoke inference and complete Opus files, not quality certification",
            "metadata": records, "manual_review_required": "Listen to both complete examples; record hardware, Stop and runtime integration separately."}, indent=2))
        return 0
    except (logic.OmniVoiceError, assets_store.AssetError, OSError, ValueError, subprocess.SubprocessError) as exc:
        code = exc.code if isinstance(exc, logic.OmniVoiceError) else str(exc) if isinstance(exc, assets_store.AssetError) else "qualification_failed"
        print(json.dumps({"status": "failed", "code": code})); return 1


if __name__ == "__main__": raise SystemExit(main())
