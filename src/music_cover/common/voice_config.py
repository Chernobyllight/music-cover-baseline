"""Shared, read-only input checks and atomic publication for voice configuration tools."""
import json
import math
import os
from pathlib import Path
import re
import tempfile
import soundfile as sf
import yaml
from ..audio import sha256
from .io import identity
from .voice_paths import portable_path

SEPARATION_TOOL = "music_cover.voice_data_separation"


def project_path(value, root):
    return (Path(root) / Path(value).expanduser()).resolve()


def check_id(value, name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{name} must be a simple ID, not a path")


def check_splits(validation, test):
    if len(validation) != len(set(validation)) or len(test) != len(set(test)):
        raise ValueError("Duplicate validation/test IDs")
    if set(validation) & set(test):
        raise ValueError("validation-ids and test-ids overlap")


def fresh_targets(targets, inputs=()):
    for name, file in targets.items():
        if file.exists() or file.is_symlink():
            raise ValueError(f"{name} already exists; choose a fresh version/path: {file}")
        for other in [*targets.values(), *inputs]:
            if file == other and other not in inputs:
                continue
            if file.is_relative_to(other) or other.is_relative_to(file):
                raise ValueError("Output destinations must not contain one another or overlap inputs")
    if len(set(targets.values())) != len(targets):
        raise ValueError("Output destinations must be distinct")


def read_json(file):
    value = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {file}")
    return value


def checked_recordings(args, root):
    directory = project_path(args.recordings_dir, root)
    manifest_file = directory / "manifest.json"
    if not manifest_file.is_file():
        raise ValueError(f"Separation manifest missing: {manifest_file}. First complete step 4 (scripts/voice_data/separate/run.sh).")
    manifest = read_json(manifest_file)
    if (manifest.get("tool") != SEPARATION_TOOL or manifest.get("schema_version") != 1
            or manifest.get("status") != "succeeded" or manifest.get("settings", {}).get("mode") != "lead"):
        raise ValueError("recordings-dir must contain a succeeded separation batch in lead mode")
    rows = manifest.get("recordings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("No recordings in separation manifest")
    seen, selected, excluded = set(), [], []
    for row in rows:
        key = row["id"]
        if not isinstance(key, str) or not key or key.startswith(".") or Path(key).name != key or "\\" in key or key in seen:
            raise ValueError(f"Unsafe/duplicate recording ID: {key!r}")
        seen.add(key)
        if row.get("excluded") is True:
            excluded.append(key)
            continue
        if row.get("excluded") is not False or row.get("status") not in {"succeeded", "skipped"}:
            raise ValueError(f"Recording is not successfully separated: {key}")
        folder = directory / key
        lead = folder / "lead.wav"
        if not lead.resolve().is_relative_to(directory) or lead.is_symlink() or folder.is_symlink():
            raise ValueError(f"Recording must stay inside recordings-dir: {key}")
        report = read_json(folder / "separation.json")
        fingerprint = identity({"source_sha256": row["source_sha256"], "settings": manifest["settings"]})
        if (report.get("tool") != SEPARATION_TOOL or report.get("status") != "succeeded"
                or row.get("identity") != fingerprint or report.get("identity") != fingerprint
                or report.get("source_sha256") != row["source_sha256"] or report.get("settings") != manifest["settings"]):
            raise ValueError(f"Separation provenance mismatch: {key}")
        digest = sha256(lead)
        if digest != report["outputs"]["lead"]["sha256"] or report["outputs"]["lead"]["path"] != "lead.wav":
            raise ValueError(f"lead.wav hash/path mismatch: {key}")
        info = sf.info(str(lead))
        if info.frames < info.samplerate or info.samplerate != 44100:
            raise ValueError(f"lead.wav must have at least one second at 44100 Hz: {key}")
        selected.append({"id": key, "path": portable_path(lead, root), "sha256": digest,
                         "source_sha256": row["source_sha256"], "duration_s": info.duration})
    held_out = set(args.validation_ids) | set(args.test_ids)
    unknown = held_out - {r["id"] for r in selected}
    if unknown:
        raise ValueError(f"Validation/test IDs are missing or excluded from this batch: {sorted(unknown)}")
    content_splits = {}
    for row in selected:
        row["split"] = ("validation" if row["id"] in args.validation_ids else
                        "test" if row["id"] in args.test_ids else "train")
        for kind in ("sha256", "source_sha256"):
            key = (kind, row[kind])
            if content_splits.get(key, row["split"]) != row["split"]:
                raise ValueError(f"Duplicate recording content across splits: {row['id']} ({kind})")
            content_splits[key] = row["split"]
    if {r["split"] for r in selected} != {"train", "validation", "test"}:
        raise ValueError("Each of train, validation and test must contain recordings")
    return selected, excluded, manifest_file


def stage_artifacts(state, stage, roles):
    record = state.get("stages", {}).get(stage, {})
    if record.get("status") != "succeeded":
        raise ValueError(f"Source run needs a successful {stage}")
    attempt = Path(record["attempt"])
    marker_file = attempt / "outputs.json"
    if sha256(marker_file) != record["outputs_manifest_sha256"]:
        raise ValueError(f"Stage outputs manifest hash mismatch: {stage}")
    marker = read_json(marker_file)
    if (marker.get("status") != "succeeded" or marker.get("stage") != stage
            or marker.get("fingerprint") != record.get("fingerprint")):
        raise ValueError(f"Stage outputs manifest differs from current state: {stage}")
    result = {}
    for role in roles:
        row = record["outputs"][role]
        saved = marker["artifacts"][role]
        file = Path(row["path"])
        if (not file.resolve().is_relative_to(attempt.resolve())
                or (attempt / saved["path"]).resolve() != file.resolve()
                or row["sha256"] != saved["sha256"] or row["bytes"] != saved["bytes"]
                or file.stat().st_size != row["bytes"] or sha256(file) != row["sha256"]):
            raise ValueError(f"Stage artifact missing or changed: {stage}/{role}")
        result[role] = file
    return result


def validation_windows(values, duration):
    result = []
    for value in values:
        try:
            start, width = map(float, value.split(":"))
        except ValueError as exc:
            raise ValueError(f"Use START:DURATION in seconds for validation-windows: {value}") from exc
        if not all(map(math.isfinite, (start, width))) or start < 0 or width <= 0 or start + width > duration:
            raise ValueError(f"Validation window {value} is invalid or exceeds source duration {duration:.3f}s")
        if any(start < w["start"] + w["duration"] and w["start"] < start + width for w in result):
            raise ValueError(f"Validation windows overlap: {value}")
        result.append({"start": start, "duration": width})
    return result


def write_new_config(output, config):
    """Publish a complete new file; never overwrite an existing path, even in a race."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".training-config-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
