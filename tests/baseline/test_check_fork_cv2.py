"""Exercise the dependency guard against the files a real patch produces."""
import contextlib
import difflib
import io
from pathlib import Path
import runpy
import tempfile
import unittest

check = runpy.run_path(str(Path(__file__).with_name('check_fork_cv2.py')))['main']


class ForkCV2GuardTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='viame-guard-test-')
        self.addCleanup(self.temporary.cleanup)
        self.packages = Path(self.temporary.name)
        self.source = self.packages / 'pytorch-libs/example'
        self.patches = self.packages / 'patches'
        self.source.mkdir(parents=True)
        self.patches.mkdir()

    def patch(self, before, after, filename='runtime.py'):
        return ''.join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True),
            fromfile='a/' + filename if before else '/dev/null',
            tofile='b/' + filename if after else '/dev/null'))

    def run_check(self, expected):
        before = {p.relative_to(self.source): p.read_bytes()
                  for p in self.source.rglob('*') if p.is_file()}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = check([str(self.packages)])
        self.assertEqual(result, expected, output.getvalue())
        after = {p.relative_to(self.source): p.read_bytes()
                 for p in self.source.rglob('*') if p.is_file()}
        self.assertEqual(after, before, 'guard modified the original fork')
        return output.getvalue()

    def test_surviving_import_fails_on_clean_and_already_patched_sources(self):
        before = 'import cv2\ndef later():\n    import cv2\n'
        after = 'def later():\n    import cv2\n'
        (self.patches / 'example.patch').write_text(self.patch(before, after))
        for source in (before, after):
            with self.subTest(already_patched=source == after):
                (self.source / 'runtime.py').write_text(source)
                self.assertIn('runtime.py:2', self.run_check(1))

    def test_complete_removal_passes_on_clean_and_already_patched_sources(self):
        before, after = 'import cv2\nx = 1\n', 'x = 1\n'
        (self.patches / 'example.patch').write_text(self.patch(before, after))
        for source in (before, after):
            with self.subTest(already_patched=source == after):
                (self.source / 'runtime.py').write_text(source)
                self.run_check(0)

    def test_added_file_is_checked(self):
        (self.source / 'runtime.py').write_text('x = 1\n')
        (self.patches / 'example.patch').write_text(
            self.patch('', 'import cv2\n', 'added.py'))
        self.assertIn('added.py:1', self.run_check(1))

    def test_deleted_file_does_not_count_as_an_import(self):
        before = 'import cv2\n'
        (self.source / 'runtime.py').write_text(before)
        (self.patches / 'example.patch').write_text(self.patch(before, ''))
        self.run_check(0)

    def test_patch_failure_cannot_pass(self):
        (self.source / 'runtime.py').write_text('x = 3\n')
        (self.patches / 'example.patch').write_text(
            self.patch('import cv2\nx = 1\n', 'x = 1\n'))
        self.assertIn('cannot apply', self.run_check(1))

    def test_overlay_is_applied_before_diff(self):
        (self.source / 'runtime.py').write_text('import cv2\nx = 1\n')
        overlay = self.patches / 'example'
        overlay.mkdir()
        before, after = 'import cv2\nx = 2\n', 'x = 2\n'
        (overlay / 'runtime.py').write_text(before)
        (self.patches / 'example.patch').write_text(self.patch(before, after))
        self.run_check(0)

    def test_dangling_documentation_asset_does_not_block_scan(self):
        (self.source / 'runtime.py').write_text('import cv2\nx = 1\n')
        (self.patches / 'example.patch').write_text(
            self.patch('import cv2\nx = 1\n', 'x = 1\n'))
        try:
            (self.source / 'logo.png').symlink_to('missing.png')
        except OSError as error:
            self.skipTest('symlinks unavailable: ' + str(error))
        self.run_check(0)

    def test_new_overlay_file_is_checked(self):
        (self.source / 'runtime.py').write_text('x = 1\n')
        overlay = self.patches / 'example'
        overlay.mkdir()
        (overlay / 'added.py').write_text('def f():\n    from cv2 import resize\n')
        self.assertIn('added.py:2', self.run_check(1))


if __name__ == '__main__':
    unittest.main()
