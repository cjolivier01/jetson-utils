import importlib
import os


def test_load_and_save_image(tmp_path=None):
    ju = importlib.import_module("jetson_utils")

    # Resolve test image path from runfiles
    img_path = os.path.join("data", "fontmapA.png")
    assert os.path.exists(img_path), f"missing test asset: {img_path}"

    # Load the image to GPU memory
    img = ju.loadImage(filename=img_path, format="rgb8")
    assert img.width > 0 and img.height > 0

    # Save it back out
    out_dir = tmp_path if tmp_path else os.environ.get("TEST_TMPDIR", ".")
    out_path = os.path.join(str(out_dir), "roundtrip.png")
    ju.saveImage(filename=out_path, image=img)
    assert os.path.exists(out_path)

