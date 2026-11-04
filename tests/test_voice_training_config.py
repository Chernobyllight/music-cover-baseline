import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import yaml

from music_cover import voice_training_config as config
from music_cover.audio import sha256
from music_cover.common.io import identity, write_json


def fixture(root):
    """A completed separator batch and a song run; no learned model is used."""
    def audio(file, frequency, seconds=2):
        file.parent.mkdir(parents=True, exist_ok=True)
        times = np.arange(round(44100 * seconds)) / 44100
        sf.write(file, 0.1 * np.sin(2 * np.pi * frequency * times), 44100)
    directory = root / "data/new/recordings"
    settings = {"mode": "lead", "sample_rate": 44100}
    manifest = {"tool": config.SEPARATION_TOOL, "schema_version": 1, "status": "succeeded",
                "settings": settings, "recordings": []}
    for index, key in enumerate(("a", "b", "c", "d")):
        source_sha = hashlib.sha256(key.encode()).hexdigest()
        fingerprint = identity({"source_sha256": source_sha, "settings": settings})
        lead = directory / key / "lead.wav"
        audio(lead, 220 + 30 * index)
        manifest["recordings"].append({"id": key, "excluded": False, "status": "succeeded",
                                       "source_sha256": source_sha, "identity": fingerprint})
        write_json(lead.parent / "separation.json", {
            "tool": config.SEPARATION_TOOL, "status": "succeeded", "settings": settings,
            "source_sha256": source_sha, "identity": fingerprint,
            "outputs": {"lead": {"path": "lead.wav", "sha256": sha256(lead)}}})
    manifest["recordings"].append({"id": "prompt", "excluded": True, "status": "excluded"})
    write_json(directory / "manifest.json", manifest)
    run = root / "runs/song"
    state = {"path_base": "run", "stages": {}}
    for stage, number, roles in (("00_prepare", 7, ("reference",)),
                                 ("04_vocal_separation", 3, ("vocals", "instrumental"))):
        attempt = run / "stages" / stage / f"attempt_{number:03d}"
        record = {"status": "succeeded", "fingerprint": stage, "attempt": str(attempt.relative_to(run)), "outputs": {}}
        marker = {"status": "succeeded", "stage": stage, "fingerprint": stage, "artifacts": {}}
        for index, role in enumerate(roles):
            file = attempt / "artifacts" / f"{role}.wav"
            audio(file, 440 + index * 110, seconds=5)
            row = {"path": str(file.relative_to(run)), "sha256": sha256(file), "bytes": file.stat().st_size}
            record["outputs"][role] = row
            marker["artifacts"][role] = dict(row, path=str(file.relative_to(attempt)))
        write_json(attempt / "outputs.json", marker)
        record["outputs_manifest_sha256"] = sha256(attempt / "outputs.json")
        state["stages"][stage] = record
    write_json(run / "stage_state.json", state)
    checkpoint, model = root / config.BASE_CHECKPOINT, root / config.BASE_CONFIG
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test placeholder, never loaded")
    model.write_text(yaml.safe_dump({"model_params": {"test_only": True}}))
    return ["--voice-id", "singer", "--version", "manual", "--recordings-dir", "data/new/recordings",
            "--source-run", "runs/song", "--validation-ids", "c", "--test-ids", "d",
            "--validation-windows", "0:1", "2:1"]


class TrainingConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.argv = fixture(self.root)
        self.args = config.parser().parse_args(self.argv)
        self.patch = patch.object(config, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def edit(self, relative, callback):
        file = self.root / relative
        value = json.loads(file.read_text())
        callback(value)
        write_json(file, value)

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() if p.is_file() else None
                for p in self.root.rglob("*")}

    def test_defaults_splits_portable_paths_and_current_attempts(self):
        value, output, summary = config.build_config(self.args)
        self.assertEqual(output, self.root / "configs/voice_training/singer_seedvc_manual.yaml")
        self.assertEqual(summary["recordings"], {"train": 2, "validation": 1, "test": 1})
        self.assertEqual(value["training"]["max_steps"], 1000)
        self.assertEqual(value["training"]["save_every"], 500)
        self.assertEqual(value["reference"], "runs/song/stages/00_prepare/attempt_007/artifacts/reference.wav")
        self.assertIn("attempt_003", value["configuration_generation"]["checked_stage_artifacts"]["vocals"]["path"])
        self.assertEqual(value["base_config"], config.BASE_CONFIG)
        self.assertEqual([x["group"] for x in value["sources"]], ["singer:a", "singer:b", "singer:c", "singer:d"])
        self.assertNotIn(str(self.root), yaml.safe_dump(value))

    def test_plan_is_read_only(self):
        before = self.snapshot()
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(config.main([*self.argv, "--plan"]), 0)
        self.assertIn("PLAN", stream.getvalue())
        self.assertEqual(before, self.snapshot())

    def test_generated_config_is_accepted_by_existing_prepare(self):
        from music_cover import voice_training
        value, output, _ = config.build_config(self.args)
        config.write_new_config(output, value)
        with patch.object(voice_training, "ROOT", self.root):
            loaded = voice_training.load_config(output)
            voice_training.prepare(loaded, self.root / "runs/train", self.root / "runs/train/out", None)
        manifest = json.loads((self.root / value["dataset_dir"] / "manifest.json").read_text())
        self.assertEqual({x["split"] for x in manifest["clips"]}, {"train", "validation", "test"})
        self.assertFalse(manifest["smoke_test"])

    def test_creation_writes_only_yaml_not_dataset_or_export(self):
        before = self.snapshot()
        with contextlib.redirect_stdout(io.StringIO()):
            config.main(self.argv)
        created_files = [p for p in self.snapshot().keys() - before.keys() if (self.root / p).is_file()]
        self.assertEqual(created_files, ["configs/voice_training/singer_seedvc_manual.yaml"])
        self.assertFalse((self.root / "data/voice_training/singer/manual/dataset").exists())
        self.assertFalse((self.root / "models/voices/singer/seed_vc/manual").exists())

    def test_cwd_and_project_relocation(self):
        original, _, _ = config.build_config(self.args)
        with tempfile.TemporaryDirectory() as directory:
            relocated = Path(directory) / "different-mount/project"
            shutil.copytree(self.root, relocated)
            old = Path.cwd()
            try:
                os.chdir(directory)
                with patch.object(config, "ROOT", relocated):
                    actual, _, _ = config.build_config(self.args)
            finally:
                os.chdir(old)
        self.assertEqual(original, actual)

    def test_explicit_paths_and_parameters(self):
        args = config.parser().parse_args([*self.argv, "--output", "configs/custom.yaml", "--dataset-dir", "data/custom",
                                          "--export-dir", "models/custom", "--max-steps", "25", "--save-every", "10",
                                          "--seed", "9", "--validation-seed", "10", "--validation-steps", "20"])
        value, output, _ = config.build_config(args)
        self.assertEqual(output, self.root / "configs/custom.yaml")
        self.assertEqual(value["dataset_dir"], "data/custom")
        self.assertEqual(value["export_dir"], "models/custom")
        self.assertEqual(value["training"]["max_steps"], 25)
        self.assertEqual(value["validation"]["seed"], 10)

    def test_no_overwrite_even_when_created_after_validation(self):
        value, output, _ = config.build_config(self.args)
        output.parent.mkdir(parents=True)
        output.write_text("existing")
        with self.assertRaises(FileExistsError):
            config.write_new_config(output, value)
        self.assertEqual(output.read_text(), "existing")
        self.assertEqual(list(output.parent.iterdir()), [output])

    def test_existing_destinations_rejected(self):
        for key, target in (("output", "configs/existing.yaml"), ("dataset_dir", "data/existing"), ("export_dir", "models/existing")):
            with self.subTest(key=key):
                file = self.root / target
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("existing")
                args = config.parser().parse_args(self.argv)
                setattr(args, key, target)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    config.build_config(args)

    def test_nested_destinations_rejected(self):
        self.args.dataset_dir, self.args.output = "new", "new/config.yaml"
        with self.assertRaisesRegex(ValueError, "must not contain"):
            config.build_config(self.args)

    def test_missing_step4_is_actionable(self):
        self.args.recordings_dir = "data/not-yet-separated"
        with self.assertRaisesRegex(ValueError, "First complete step 4"):
            config.build_config(self.args)

    def test_failed_batch_rejected_even_when_old_leads_exist(self):
        self.edit("data/new/recordings/manifest.json", lambda value: value.update(status="failed"))
        with self.assertRaisesRegex(ValueError, "succeeded separation batch"):
            config.build_config(self.args)

    def test_verified_resumed_recording_is_accepted(self):
        self.edit("data/new/recordings/manifest.json", lambda value: value["recordings"][0].update(status="skipped"))
        self.assertEqual(config.build_config(self.args)[2]["recordings"]["train"], 2)

    def test_changed_lead_rejected(self):
        with (self.root / "data/new/recordings/a/lead.wav").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash/path mismatch"):
            config.build_config(self.args)

    def test_provenance_mismatch_rejected(self):
        self.edit("data/new/recordings/a/separation.json", lambda value: value.update(identity="old"))
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            config.build_config(self.args)

    def test_duplicate_recording_id_rejected(self):
        self.edit("data/new/recordings/manifest.json", lambda value: value["recordings"].append(value["recordings"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate recording ID"):
            config.build_config(self.args)

    def test_split_id_errors(self):
        for validation, test, error in ((["missing"], ["d"], "missing or excluded"), (["prompt"], ["d"], "missing or excluded"),
                                         (["c", "c"], ["d"], "Duplicate"), (["c"], ["c"], "overlap"),
                                         (["a", "b", "c"], ["d"], "Each of train")):
            with self.subTest(validation=validation, test=test):
                self.args.validation_ids, self.args.test_ids = validation, test
                with self.assertRaisesRegex(ValueError, error):
                    config.build_config(self.args)

    def test_duplicate_audio_across_splits_rejected(self):
        original = self.root / "data/new/recordings/a/lead.wav"
        shutil.copyfile(original, self.root / "data/new/recordings/c/lead.wav")
        self.edit("data/new/recordings/c/separation.json", lambda value: value["outputs"]["lead"].update(sha256=sha256(original)))
        with self.assertRaisesRegex(ValueError, "Duplicate recording content across splits"):
            config.build_config(self.args)

    def test_changed_stage_marker_rejected(self):
        file = self.root / "runs/song/stages/04_vocal_separation/attempt_003/outputs.json"
        file.write_text(file.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            config.build_config(self.args)

    def test_changed_stage_audio_rejected(self):
        file = self.root / "runs/song/stages/04_vocal_separation/attempt_003/artifacts/vocals.wav"
        file.write_bytes(file.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
            config.build_config(self.args)

    def test_failed_stage_not_replaced_by_older_attempt(self):
        self.edit("runs/song/stage_state.json", lambda value: value["stages"]["04_vocal_separation"].update(status="failed"))
        with self.assertRaisesRegex(ValueError, "successful 04_vocal_separation"):
            config.build_config(self.args)

    def test_bad_windows(self):
        for windows in (("nan:1",), ("0:inf",), ("-1:1",), ("0:0",), ("4:2",), ("wrong",), ("0:2", "1:2")):
            with self.subTest(windows=windows):
                self.args.validation_windows = windows
                with self.assertRaises(ValueError):
                    config.build_config(self.args)

    def test_invalid_parameters(self):
        for key, value in (("voice_id", "../other"), ("version", "../other"), ("max_steps", 0), ("learning_rate", float("nan")),
                           ("segment_seconds", float("inf")), ("gpu", -1), ("seed", -1), ("validation_steps", 0)):
            with self.subTest(key=key):
                args = config.parser().parse_args(self.argv)
                setattr(args, key, value)
                with self.assertRaises(ValueError):
                    config.build_config(args)

    def test_missing_base_checkpoint(self):
        (self.root / config.BASE_CHECKPOINT).unlink()
        with self.assertRaisesRegex(ValueError, "Base model asset missing"):
            config.build_config(self.args)

    def test_task_yaml_cannot_be_used_as_model_architecture(self):
        (self.root / config.BASE_CONFIG).write_text("voice_id: old_experiment\n")
        with self.assertRaisesRegex(ValueError, "model architecture YAML"):
            config.build_config(self.args)


if __name__ == "__main__":
    unittest.main()
