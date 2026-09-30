import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from data_loader.normal import FILENAME_ORDERED_FPS, get_eval_items, scan_dirs


class NormalScanDirsTests(unittest.TestCase):
    def _make_dir(self, root: Path, names: list[str]) -> Path:
        image_dir = root / "frames"
        image_dir.mkdir()
        for name in names:
            (image_dir / name).touch()
        return image_dir

    def test_legacy_names_keep_time_order_and_no_fallback_fps(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_dir = self._make_dir(
                Path(temp_dir), ["1_2.jpg", "0_10.png", "0_2.jpeg"]
            )

            _pairs, all_images = scan_dirs([str(image_dir)])

            items = all_images[str(image_dir)]
            self.assertEqual(
                [Path(item.image_path).name for item in items],
                ["0_2.jpeg", "0_10.png", "1_2.jpg"],
            )
            self.assertEqual(
                [(item.second, item.frame) for item in items],
                [(0, 2), (0, 10), (1, 2)],
            )
            self.assertTrue(all(item.fps is None for item in items))
            self.assertEqual(
                [item.output_stem for item in items], ["0_2", "0_10", "1_2"]
            )

    def test_arbitrary_names_use_name_order_and_30_fps(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_dir = self._make_dir(
                Path(temp_dir),
                ["frame10.png", "frame2.JPG", "0_0.jpg", "ignored.txt"],
            )

            _pairs, all_images = scan_dirs([str(image_dir)])

            items = all_images[str(image_dir)]
            self.assertEqual(
                [Path(item.image_path).name for item in items],
                ["0_0.jpg", "frame10.png", "frame2.JPG"],
            )
            self.assertEqual([item.frame for item in items], [0, 1, 2])
            self.assertTrue(all(item.second == 0 for item in items))
            self.assertTrue(all(item.fps == FILENAME_ORDERED_FPS for item in items))
            self.assertEqual(
                [item.output_stem for item in items],
                ["0_0", "frame10", "frame2"],
            )

    def test_empty_image_directory_is_omitted(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_dir = self._make_dir(Path(temp_dir), [])

            _pairs, all_images = scan_dirs([str(image_dir)])

            self.assertNotIn(str(image_dir), all_images)

    def test_eval_rejects_case_insensitive_stem_collisions(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_dir = self._make_dir(Path(temp_dir), ["Photo.jpg", "photo.png"])

            with self.assertRaisesRegex(ValueError, "Duplicate image stem") as raised:
                get_eval_items([str(image_dir)])

            self.assertIn(str(image_dir), str(raised.exception))

    def test_eval_rejects_colliding_normalized_legacy_stems(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_dir = self._make_dir(Path(temp_dir), ["1_2.jpg", "01_02.png"])

            with self.assertRaisesRegex(ValueError, "Duplicate eval output stem") as raised:
                get_eval_items([str(image_dir)])

            self.assertIn(str(image_dir), str(raised.exception))


if __name__ == "__main__":
    unittest.main()
