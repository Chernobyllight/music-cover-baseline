"""File safety, resume and backend-contract tests; no model weights are loaded."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from music_cover import voice_data_separation as module


class SeparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.patch = patch.object(module, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.source = self.root / "raw"
        for key in ("a", "b", "reference"):
            target = self.source / key / "original.wav"
            target.parent.mkdir(parents=True)
            sf.write(target, .2 * np.sin(np.arange(44100) * .04), 44100, subtype="FLOAT")
        self.args = module.parser().parse_args([
            "--input-dir", "raw", "--pattern", "*/original.wav", "--output-dir", "prepared",
            "--exclude-ids", "reference"])
        self.settings = {"mode": "lead", "model_assets": {"weights": "first"}}
        self.calls = []

    def fake_runner(self, request, work, log):
        self.calls.append(request["source"])
        for role in module.ROLES[request["settings"]["mode"]]:
            shutil.copy2(request["source"], work / f"{role}.wav")

    def run_batch(self, runner=None):
        _, output, rows = module.discover(self.args)
        with contextlib.redirect_stdout(io.StringIO()):
            return module.run_batch(self.args, output, rows, self.settings, runner or self.fake_runner)

    def test_discovery_is_relative_to_project_not_cwd_and_excludes_prompt(self):
        previous = Path.cwd()
        try:
            os.chdir("/")
            _, output, rows = module.discover(self.args)
        finally:
            os.chdir(previous)
        self.assertEqual(output, self.root / "prepared")
        self.assertEqual([r["id"] for r in rows if not r["excluded"]], ["a", "b"])
        self.assertEqual(rows[0]["source"], "raw/a/original.wav")
        self.assertFalse(output.exists())

    def test_rejects_overlap_unknown_exclusion_and_colliding_ids(self):
        self.args.output_dir = "raw/inside"
        with self.assertRaisesRegex(ValueError, "contain"):
            module.discover(self.args)
        self.args.output_dir = "prepared"
        self.args.exclude_ids = ["typo"]
        with self.assertRaisesRegex(ValueError, "do not match"):
            module.discover(self.args)
        self.args.exclude_ids = []
        shutil.copy2(self.source / "a/original.wav", self.source / "a/another.wav")
        self.args.pattern = "*/*.wav"
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            module.discover(self.args)

    def test_full_batch_and_resume_verify_outputs(self):
        self.assertEqual(self.run_batch(), 0)
        self.assertEqual(len(self.calls), 2)
        self.args.resume = True
        self.assertEqual(self.run_batch(), 0)
        self.assertEqual(len(self.calls), 2)
        report = json.loads((self.root / "prepared/manifest.json").read_text())
        self.assertEqual(report["counts"], {"skipped": 2, "excluded": 1})
        self.assertFalse((self.root / "prepared/reference").exists())
        info = json.loads((self.root / "prepared/a/separation.json").read_text())
        self.assertEqual(info["outputs"]["lead"]["sample_rate"], 44100)

    def test_missing_or_corrupted_output_is_never_skipped(self):
        self.run_batch()
        self.args.resume = True
        (self.root / "prepared/a/lead.wav").unlink()
        (self.root / "prepared/b/lead.wav").write_bytes(b"corrupt")
        self.assertEqual(self.run_batch(), 1)
        report = json.loads((self.root / "prepared/manifest.json").read_text())
        self.assertEqual(report["counts"]["failed"], 2)

    def test_input_or_model_change_invalidates_resume(self):
        self.run_batch()
        self.args.resume = True
        changed = self.source / "a/original.wav"
        sf.write(changed, np.ones(44100) * .3, 44100, subtype="FLOAT")
        self.assertEqual(self.run_batch(), 1)
        self.settings["model_assets"]["weights"] = "changed"
        self.assertEqual(self.run_batch(), 1)
        report = json.loads((self.root / "prepared/manifest.json").read_text())
        self.assertEqual(report["counts"]["failed"], 2)

    def test_force_archives_previous_audio_without_deleting_it(self):
        self.run_batch()
        before = (self.root / "prepared/a/lead.wav").read_bytes()
        self.args.force = True
        self.settings["model_assets"]["weights"] = "changed"
        self.assertEqual(self.run_batch(), 0)
        backups = list((self.root / "prepared/.history").glob("a-*/lead.wav"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), before)

    def test_failed_worker_leaves_no_trainable_partial_and_resume_retries(self):
        def failing(request, work, log):
            if "/a/" in request["source"]:
                shutil.copy2(request["source"], work / "lead.wav")
                raise RuntimeError("fake GPU failure")
            self.fake_runner(request, work, log)
        self.assertEqual(self.run_batch(failing), 1)
        self.assertFalse((self.root / "prepared/a").exists())
        self.assertTrue(list((self.root / "prepared/.work").glob("a-*/failure.json")))
        self.args.resume = True
        self.assertEqual(self.run_batch(), 0)
        self.assertEqual(len(self.calls), 2)

    def test_prevents_unintended_overwrite_and_changed_dataset_selection(self):
        self.run_batch()
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.run_batch()
        self.args.resume = True
        self.args.exclude_ids.append("a")
        with self.assertRaisesRegex(ValueError, "another recording selection"):
            self.run_batch()

    def test_plan_writes_nothing_and_does_not_call_worker(self):
        with patch.object(module, "preflight", return_value=self.settings), \
             patch.object(module, "run_batch") as execute, contextlib.redirect_stdout(io.StringIO()):
            result = module.main(["--input-dir", "raw", "--pattern", "*/original.wav",
                                  "--output-dir", "prepared", "--plan"])
        self.assertEqual(result, 0)
        execute.assert_not_called()
        self.assertFalse((self.root / "prepared").exists())

    def test_invalid_audio_or_duration_cannot_be_published(self):
        def invalid(request, work, log):
            self.fake_runner(request, work, log)
            sf.write(work / "lead.wav", np.zeros(10), 22050, subtype="FLOAT")
        self.assertEqual(self.run_batch(invalid), 1)
        self.assertFalse(list((self.root / "prepared").glob("*/lead.wav")))

    def test_worker_matches_two_stage_contract_without_loading_models(self):
        work = self.root / "worker"; work.mkdir()
        models = []; options = []
        class FakeSeparator:
            def __init__(self, **kwargs): options.append(kwargs)
            def load_model(self, model_filename): models.append(model_filename)
            def separate(self, source, names):
                for name in names.values():
                    shutil.copy2(source, Path(options[-1]["output_dir"]) / (name + ".wav"))
        request = work / "worker_request.json"
        module.write_json(request, {"gpu": 0, "source": str(self.source / "a/original.wav"),
            "model_dir": str(self.root / "models"), "settings": {
                "mode": "lead", "vocals_model": module.VOCALS_MODEL, "lead_model": module.LEAD_MODEL,
                "vocals_overlap": 8, "lead_overlap": 4}})
        fake = types.ModuleType("audio_separator.separator")
        fake.Separator = FakeSeparator
        previous = Path.cwd()
        try:
            with patch.dict("sys.modules", {"audio_separator.separator": fake}):
                module.worker(str(request))
        finally:
            os.chdir(previous)
        self.assertEqual(models, [module.VOCALS_MODEL, module.LEAD_MODEL])
        self.assertEqual([o["mdxc_params"]["overlap"] for o in options], [8, 4])
        self.assertTrue(all(o["sample_rate"] == 44100 and o["use_soundfile"] for o in options))
        self.assertTrue((work / "lead.wav").is_file())

    def test_subprocess_uses_current_base_python_and_gpu(self):
        work = self.root / "worker"; work.mkdir()
        request = {"gpu": 2, "timeout_seconds": 15}
        with patch.object(module.subprocess, "run") as execute, (work / "log").open("w") as log:
            module.execute_worker(request, work, log)
        args, kwargs = execute.call_args
        self.assertEqual(args[0][0], module.sys.executable)
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(kwargs["cwd"], work)


if __name__ == "__main__":
    unittest.main()
