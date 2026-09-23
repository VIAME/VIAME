# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add vertex-ai plugin directory to sys.path
PLUGIN_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "plugins", "vertex-ai")
)
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

from train_handler import TrainHandler


class TestTrainHandler(unittest.TestCase):

    def setUp(self):
        self.viame_dir = "/tmp/test_viame"
        self.work_dir = "/tmp/test_work"
        self.handler = TrainHandler(
            viame_dir=self.viame_dir,
            work_dir=self.work_dir
        )

    @patch("train_handler.subprocess.Popen")
    @patch.object(TrainHandler, "_get_viame_env")
    def test_normal_command_construction(self, mock_get_env, mock_popen):
        """1. Verify normal command construction works with Python list arguments."""
        mock_get_env.return_value = {"PATH": "/opt/viame/bin"}
        mock_proc = MagicMock()
        mock_proc.stdout = ["Training finished"]
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        payload = {
            "input_dir": "/data/input",
            "config": "detector.conf",
            "settings": {"lr": "0.001"}
        }

        result = self.handler.run(payload)

        self.assertTrue(mock_popen.called)
        args, kwargs = mock_popen.call_args
        cmd_list = args[0]

        expected_cmd = [
            "viame", "train",
            "-c", "detector.conf",
            "-i", "/data/input",
            "-o", os.path.join(self.work_dir, "training_output"),
            "-s", "lr=0.001"
        ]
        self.assertEqual(cmd_list, expected_cmd)
        self.assertEqual(kwargs.get("env"), {"PATH": "/opt/viame/bin"})
        self.assertFalse(kwargs.get("shell", False))
        self.assertEqual(result["status"], "completed")

    @patch("train_handler.subprocess.Popen")
    @patch.object(TrainHandler, "_get_viame_env")
    def test_arguments_with_spaces_remain_single_arguments(self, mock_get_env, mock_popen):
        """2. Verify arguments containing spaces remain single elements in the argv list."""
        mock_get_env.return_value = {}
        mock_proc = MagicMock()
        mock_proc.stdout = []
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        payload = {
            "input_dir": "/path with spaces/input data",
            "config": "my custom config.conf",
            "settings": {"option name": "value with spaces"}
        }

        self.handler.run(payload)

        args, kwargs = mock_popen.call_args
        cmd_list = args[0]

        self.assertFalse(kwargs.get("shell", False))
        self.assertEqual(cmd_list[cmd_list.index("-i") + 1], "/path with spaces/input data")
        self.assertEqual(cmd_list[cmd_list.index("-c") + 1], "my custom config.conf")
        self.assertIn("option name=value with spaces", cmd_list)

    @patch("train_handler.subprocess.Popen")
    @patch.object(TrainHandler, "_get_viame_env")
    def test_shell_metacharacters_passed_literally_not_executed(self, mock_get_env, mock_popen):
        """3. Verify shell metacharacters are passed as literal elements in argv with shell=False (Popen mocked)."""
        mock_get_env.return_value = {}
        mock_proc = MagicMock()
        mock_proc.stdout = []
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        marker_file = os.path.join(self.work_dir, "injected_marker.txt")
        malicious_input = f"dir; touch {marker_file}"

        payload = {
            "input_dir": malicious_input,
            "config": "$(reboot)",
            "settings": {"key|pipe": "val && rm -rf /"}
        }

        self.handler.run(payload)

        args, kwargs = mock_popen.call_args
        cmd_list = args[0]

        self.assertIsInstance(cmd_list, list)
        self.assertFalse(kwargs.get("shell", False))
        self.assertIn(malicious_input, cmd_list)
        self.assertIn("$(reboot)", cmd_list)
        self.assertIn("key|pipe=val && rm -rf /", cmd_list)

    @patch("train_handler.subprocess.Popen")
    @patch.object(TrainHandler, "_get_viame_env")
    def test_subprocess_failure_resets_status(self, mock_get_env, mock_popen):
        """4. Verify subprocess failure resets the handler status correctly."""
        mock_get_env.return_value = {}
        mock_proc = MagicMock()
        mock_proc.stdout = ["Error encountered"]
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        result = self.handler.run({"input_dir": "/data"})

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["return_code"], 1)
        self.assertEqual(self.handler.get_status()["state"], "failed")

    def test_missing_setup_script_raises_runtime_error(self):
        """5. Verify missing setup script raises a clear RuntimeError."""
        handler = TrainHandler("/nonexistent_dir", self.work_dir)
        with self.assertRaises(RuntimeError) as ctx:
            handler._get_viame_env()
        self.assertIn("VIAME setup script not found", str(ctx.exception))

    @patch("train_handler.subprocess.run")
    def test_setup_script_failure_raises_runtime_error_with_stderr(self, mock_run):
        """6. Verify setup script sourcing failure raises RuntimeError with stderr output."""
        mock_proc = MagicMock()
        mock_proc.returncode = 127
        mock_proc.stderr = "setup_viame.sh: syntax error line 5"
        mock_run.return_value = mock_proc

        with patch("os.path.isfile", return_value=True):
            handler = TrainHandler("/fake/viame", self.work_dir)
            with self.assertRaises(RuntimeError) as ctx:
                handler._get_viame_env()
            self.assertIn("Failed to source", str(ctx.exception))
            self.assertIn("setup_viame.sh: syntax error line 5", str(ctx.exception))

    @patch.object(TrainHandler, "_get_viame_env")
    def test_preparation_failure_resets_status_to_failed(self, mock_get_env):
        """7. Verify handler status becomes 'failed' when setup/environment preparation raises an exception."""
        mock_get_env.side_effect = RuntimeError("Failed to prepare environment")

        with self.assertRaises(RuntimeError):
            self.handler.run({"input_dir": "/data"})

        self.assertEqual(self.handler.get_status()["state"], "failed")


if __name__ == "__main__":
    unittest.main()
