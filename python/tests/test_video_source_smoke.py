import importlib
import os


def test_video_source_from_image():
    ju = importlib.import_module("jetson_utils")

    # Use a static image file as a source
    img_path = os.path.join("data", "fontmapA.png")
    assert os.path.exists(img_path)
    uri = f"file://{img_path}"

    src = ju.videoSource(uri, argv=["test"])  # positional argv is fine
    frame = src.Capture()
    assert frame is not None
    assert frame.width > 0 and frame.height > 0

