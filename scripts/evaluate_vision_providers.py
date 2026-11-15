"""Explicitly submit user-reviewed image cases to both Gemini and OpenAI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import mimetypes
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant.evaluation import EvaluationCase, new_evaluation_report, run_case_for_providers
from src.assistant.vision import (
    ImageInput,
    ImageValidationError,
    resolve_provider_config,
    validate_uploaded_image,
)


class _LocalUpload:
    def __init__(self, path: Path) -> None:
        self.name = path.name
        self.type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._data = path.read_bytes()

    def getvalue(self) -> bytes:
        return self._data


def _load_cases(manifest_path: Path) -> list[EvaluationCase]:
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read a valid evaluation manifest.") from exc
    rows = document.get("cases") if isinstance(document, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation manifest must contain a non-empty 'cases' array.")
    cases: list[EvaluationCase] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Case {index} must be an object.")
        case_id = str(row.get("case_id", "")).strip()
        question = str(row.get("question", "")).strip()
        image_rows = row.get("images")
        if not case_id or not question or not isinstance(image_rows, list) or not image_rows:
            raise ValueError(f"Case {index} needs case_id, question, and at least one image.")
        images: list[ImageInput] = []
        for image_index, image_row in enumerate(image_rows, start=1):
            if not isinstance(image_row, dict) or not str(image_row.get("path", "")).strip():
                raise ValueError(f"Case {case_id} image {image_index} needs a local path.")
            raw_path = Path(str(image_row["path"]))
            path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
            try:
                image = validate_uploaded_image(_LocalUpload(path.resolve(strict=True)))
            except (OSError, ImageValidationError) as exc:
                raise ValueError(f"Case {case_id} image {image_index} is not a readable PNG/JPEG file.") from exc
            label = str(image_row.get("label", "")).strip() or image.filename
            images.append(ImageInput(
                image.image_id, image.filename, image.mime_type, image.data,
                image.width, image.height, label,
            ))
        context = row.get("engineering_context", {})
        if not isinstance(context, dict) or any(not isinstance(value, str) for value in context.values()):
            raise ValueError(f"Case {case_id} engineering_context must map names to text values.")
        cases.append(EvaluationCase(
            case_id=case_id,
            question=question,
            images=tuple(images),
            engineering_context=context,
            comparison_requested=bool(row.get("comparison_requested", False)),
            reviewer_notes=str(row.get("reviewer_notes", "")),
        ))
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare configured Gemini and OpenAI vision providers using the same reviewed image cases."
    )
    parser.add_argument("manifest", type=Path, help="JSON manifest containing local, non-confidential test images")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path; existing files are never overwritten")
    parser.add_argument(
        "--confirm-external-upload", action="store_true",
        help="Explicitly authorize sending every manifest image to both named providers",
    )
    args = parser.parse_args(argv)
    if not args.confirm_external_upload:
        parser.error("External uploads are disabled by default; review provider privacy and pass --confirm-external-upload to proceed.")

    output_path = args.output.resolve()
    if output_path.exists():
        print("Output file already exists; choose a new path. No images were sent and no report was written.", file=sys.stderr)
        return 2

    gemini = resolve_provider_config(provider="gemini")
    openai = resolve_provider_config(provider="openai")
    configs = [gemini, openai]
    if not all(config.is_configured for config in configs):
        print("Comparison requires both GEMINI_API_KEY and OPENAI_API_KEY with their configured model IDs.", file=sys.stderr)
        return 2
    try:
        cases = _load_cases(args.manifest.resolve(strict=True))
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if any(len(case.images) > config.max_images for case in cases for config in configs):
        print("A case exceeds one provider's configured image-count limit; no images were sent.", file=sys.stderr)
        return 2

    report = new_evaluation_report(
        dataset_label="User-supplied reviewed cases; not a verified benchmark",
        providers=configs,
        cases=cases,
    )
    for report_case, case in zip(report["cases"], cases):
        report_case["results"] = run_case_for_providers(case, configs)
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()

    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as output_file:
            json.dump(report, output_file, indent=2, ensure_ascii=False)
            output_file.write("\n")
    except OSError as exc:
        print("Could not create the requested evaluation report.", file=sys.stderr)
        return 2
    print(f"Evaluation completed. Review provider outputs and fill manual rubric scores in: {output_path}")
    print("The report contains provider-generated analysis and image filenames, but no image bytes or API keys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
