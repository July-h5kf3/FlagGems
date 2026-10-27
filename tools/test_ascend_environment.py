# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SDK_ENV = ROOT / "src/flag_gems/runtime/backend/_ascend/env.sh"


class AscendEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.sdk = Path(self.directory.name) / "sdk with spaces"
        (self.sdk / "bin").mkdir(parents=True)
        info_dir = self.sdk / f"{platform.machine()}-linux"
        info_dir.mkdir()
        self.info = info_dir / "ascend_toolkit_install.info"
        self.info.write_text("version=9.1.0\n")
        for name in ("bishengir-opt", "bishengir-compile"):
            compiler = self.sdk / "bin" / name
            compiler.write_text("#!/bin/sh\nexit 0\n")
            compiler.chmod(0o755)
        self.script = self.sdk / "set_env.sh"
        self.script.write_text(
            f"export ASCEND_HOME_PATH={shlex.quote(str(self.sdk))}\n"
            f"export PATH={shlex.quote(str(self.sdk / 'bin'))}:$PATH\n"
        )
        self.env = dict(os.environ)
        self.env.update(
            CANN_ENV_SCRIPT=str(self.script),
            ASCEND_HOME_PATH="/stale/cann-9.0",
            PYTHONPATH=str(ROOT / "src"),
        )

    def test_select_complete_sdk(self):
        for version in ("9.1.0", "9.1.1"):
            with self.subTest(version=version):
                self.info.write_text(f"version={version}\n")
                result = subprocess.run(
                    ["bash", "-ec", 'source "$1"', "bash", str(SDK_ENV)],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"CANN {version}: {self.sdk}", result.stdout)

    def test_old_sdk_fails_before_package_install(self):
        self.info.write_text("version=9.0.0\n")
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "src/flaggems_setup.py"),
                "ascend-cann910",
                "--pip",
                "must-not-run-pip",
            ],
            env=self.env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires CANN >= 9.1.0", result.stderr)
        self.assertNotIn("[Step 1]", result.stdout)

    def test_required_activation_propagates_sdk_error(self):
        self.info.write_text("version=9.0.0\n")
        result = subprocess.run(
            [
                "bash",
                str(ROOT / "tools/source_env_bash.sh"),
                str(SDK_ENV),
                "--required",
            ],
            env=self.env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("requires CANN >= 9.1.0", result.stderr)

    def test_optional_environment_keeps_existing_behavior(self):
        self.script.write_text("export LEGACY_ENV_PROBE=present\nfalse\n")
        result = subprocess.run(
            ["bash", str(ROOT / "tools/source_env_bash.sh"), str(self.script)],
            env=self.env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LEGACY_ENV_PROBE=present", result.stdout)

    def test_mixed_compiler_path_is_rejected(self):
        self.script.write_text(
            f"export ASCEND_HOME_PATH={shlex.quote(str(self.sdk))}\n"
        )
        result = subprocess.run(
            ["bash", "-ec", 'source "$1"', "bash", str(SDK_ENV)],
            env=self.env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must come from the selected CANN SDK", result.stderr)

    def test_dry_run_does_not_require_installed_sdk(self):
        self.env["CANN_ENV_SCRIPT"] = "/missing/sdk/set_env.sh"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "src/flaggems_setup.py"),
                "ascend-cann910",
                "--dry-run",
                "--pip",
                "must-not-run-pip",
            ],
            env=self.env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Required SDK environment", result.stdout)

    def test_ci_uses_configured_profile(self):
        cfg = yaml.safe_load((ROOT / "src/flag_gems/backends.yaml").read_text())
        backends = json.loads((ROOT / ".github/backends.json").read_text())
        ascend = next(backend for backend in backends if backend["vendor"] == "ascend")
        self.assertEqual(ascend["backend"], "ascend-cann910")
        self.assertTrue(cfg["backends"][ascend["backend"]]["required_env_source"])


if __name__ == "__main__":
    unittest.main()
