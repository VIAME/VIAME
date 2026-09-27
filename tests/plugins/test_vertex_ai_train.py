# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

PLUGIN_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "plugins", "vertex-ai")
)
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

from train_handler import TrainHandler


class TestTrainHandler(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="viame-train-handler-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        # Exercise quoting of the setup-script and working-directory paths too.
        self.viame_dir = root / "viame install"
        self.work_dir = root / "work directory"
        self.bin_dir = self.viame_dir / "bin"
        self.bin_dir.mkdir(parents=True)
        self.work_dir.mkdir()
        self.record = self.work_dir / "arguments.json"
        self.setup = self.viame_dir / "setup_viame.sh"
        self.setup.write_text(
            'export PATH=' + shlex.quote(str(self.bin_dir)) + ':"$PATH"\n'
            'export PR306_SETUP_VALUE="from setup"\n'
            'unset PR306_REMOVED_VARIABLE\n'
        )
        # A real subprocess records only test arguments and test variables.
        # No training, cloud access, or environment dump is needed.
        stub = self.bin_dir / "viame"
        stub.write_text(
            '#!' + sys.executable + '\n'
            'import json, os, sys\n'
            'with open(' + repr(str(self.record)) + ', "w") as out:\n'
            '    json.dump({"args": sys.argv[1:], "cwd": os.getcwd(),\n'
            '               "setup_value": os.environ.get("PR306_SETUP_VALUE"),\n'
            '               "removed_present": "PR306_REMOVED_VARIABLE" in os.environ}, out)\n'
        )
        stub.chmod(0o700)
        self.handler = TrainHandler(str(self.viame_dir), str(self.work_dir))

    def mock_process(self, mock_popen, returncode=0):
        proc = MagicMock()
        proc.stdout = ["Training finished"]
        proc.returncode = returncode
        mock_popen.return_value = proc
        return proc

    @patch("train_handler.subprocess.Popen")
    def test_normal_command_construction(self, mock_popen):
        self.mock_process(mock_popen)
        result = self.handler.run({
            "input_dir": "/data/input",
            "config": "detector.conf",
            "settings": {"lr": "0.001"},
        })
        args, kwargs = mock_popen.call_args
        self.assertEqual(args[0], [
            "bash", "-c", 'source "$1" && shift && exec "$@"',
            "--", str(self.setup), "viame", "train",
            "-c", "detector.conf", "-i", "/data/input",
            "-o", str(self.work_dir / "training_output"), "-s", "lr=0.001",
        ])
        self.assertFalse(kwargs["shell"])
        self.assertNotIn("env", kwargs)
        self.assertEqual(result["status"], "completed")

    def test_arguments_with_spaces_remain_single_arguments(self):
        result = self.handler.run({
            "input_dir": "/path with spaces/input data",
            "config": "my custom config.conf",
            "settings": {"option name": "value with spaces"},
        })
        args = json.loads(self.record.read_text())["args"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(args[args.index("-i") + 1], "/path with spaces/input data")
        self.assertEqual(args[args.index("-c") + 1], "my custom config.conf")
        self.assertIn("option name=value with spaces", args)

    def test_shell_metacharacters_are_literal_in_real_subprocess(self):
        marker = self.work_dir / "injected-marker"
        quoted_marker = shlex.quote(str(marker))
        values = [
            "data; printf injected > " + quoted_marker + "; #",
            "$(printf injected > " + quoted_marker + ")",
            "`printf injected > " + quoted_marker + "`",
            "data\nprintf injected > " + quoted_marker,
            "data | printf injected > " + quoted_marker,
            "data && printf injected > " + quoted_marker,
            "quotes ' \" and $HOME and *.png",
        ]
        for value in values:
            with self.subTest(value=value):
                result = self.handler.run({
                    "input_dir": value, "config": value, "settings": {value: value},
                })
                args = json.loads(self.record.read_text())["args"]
                self.assertEqual(result["status"], "completed")
                self.assertEqual(args[args.index("-i") + 1], value)
                self.assertEqual(args[args.index("-c") + 1], value)
                self.assertIn(value + "=" + value, args)
                self.assertFalse(marker.exists())

    def test_setup_output_is_logged_without_corrupting_environment(self):
        self.setup.write_text(
            'echo "VIAME setup banner"\necho "setup warning" >&2\n'
            + self.setup.read_text()
        )
        result = self.handler.run({"input_dir": "/data"})
        record = json.loads(self.record.read_text())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(record["setup_value"], "from setup")
        self.assertEqual(record["cwd"], str(self.work_dir))
        self.assertIn("VIAME setup banner", self.handler.get_status()["log_tail"])
        self.assertIn("setup warning", self.handler.get_status()["log_tail"])

    def test_variables_unset_by_setup_stay_unset(self):
        with patch.dict(os.environ, {"PR306_REMOVED_VARIABLE": "parent value"}):
            result = self.handler.run({"input_dir": "/data"})
        self.assertEqual(result["status"], "completed")
        self.assertFalse(json.loads(self.record.read_text())["removed_present"])

    def test_setup_runs_again_for_each_job(self):
        self.handler.run({"input_dir": "/data"})
        self.setup.write_text(self.setup.read_text().replace("from setup", "updated setup"))
        self.handler.run({"input_dir": "/data"})
        self.assertEqual(json.loads(self.record.read_text())["setup_value"], "updated setup")

    @patch("train_handler.subprocess.Popen")
    def test_subprocess_failure_resets_status(self, mock_popen):
        self.mock_process(mock_popen, returncode=1)
        result = self.handler.run({"input_dir": "/data"})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["return_code"], 1)
        self.assertEqual(self.handler.get_status()["state"], "failed")

    @patch("train_handler.subprocess.Popen")
    def test_missing_setup_script_does_not_start_training(self, mock_popen):
        self.setup.unlink()
        with self.assertRaisesRegex(RuntimeError, "VIAME setup script not found"):
            self.handler.run({"input_dir": "/data"})
        mock_popen.assert_not_called()
        self.assertEqual(self.handler.get_status()["state"], "failed")

    def test_setup_failure_is_logged_and_does_not_start_training(self):
        self.setup.write_text('echo "setup failed" >&2\nreturn 23\n')
        result = self.handler.run({"input_dir": "/data"})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["return_code"], 23)
        self.assertIn("setup failed", result["log_tail"])
        self.assertFalse(self.record.exists())
        self.assertEqual(self.handler.get_status()["state"], "failed")

    @patch("train_handler.subprocess.Popen")
    @patch.object(TrainHandler, "_resolve_gcs_path")
    def test_preparation_failure_resets_status(self, mock_resolve, mock_popen):
        mock_resolve.side_effect = RuntimeError("Failed to prepare input")
        with self.assertRaisesRegex(RuntimeError, "Failed to prepare input"):
            self.handler.run({"input_dir": "gs://test-bucket/data"})
        mock_popen.assert_not_called()
        self.assertEqual(self.handler.get_status()["state"], "failed")

    @patch.object(TrainHandler, "_upload_to_gcs")
    def test_successful_training_still_uploads_output(self, mock_upload):
        result = self.handler.run({"input_dir": "/data", "output_dir": "gs://test-bucket/models"})
        self.assertEqual(result["status"], "completed")
        mock_upload.assert_called_once_with(
            str(self.work_dir / "training_output"), "gs://test-bucket/models",
        )


if __name__ == "__main__":
    unittest.main()
