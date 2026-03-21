import importlib
import os


def test_cuda_convert_and_resize(tmp_path=None):
    ju = importlib.import_module("jetson_utils")

    # Small test image
    w, h = 8, 6

    # Allocate a mapped RGB input and RGBA output with same dims for color convert
    src = ju.cudaAllocMapped(width=w, height=h, format="rgb8")
    dst_rgba = ju.cudaAllocMapped(width=w, height=h, format="rgba8")
    ju.cudaConvertColor(src, dst_rgba)

    # Resize to a different size in-place to another buffer
    w2, h2 = 5, 4
    dst_resized = ju.cudaAllocMapped(width=w2, height=h2, format="rgba8")
    ju.cudaResize(dst_rgba, dst_resized)
    assert int(dst_resized.width) == w2 and int(dst_resized.height) == h2


def test_save_image_rgba(tmp_path):
    ju = importlib.import_module("jetson_utils")

    w, h = 4, 3
    # Allocate a float4 mapped buffer and save as PNG
    buf = ju.cudaAllocMapped(width=w, height=h, format="rgba32f")
    out_path = tmp_path / "rgba.png"
    ju.saveImageRGBA(filename=str(out_path), image=buf, width=w, height=h, max_pixel=1.0)
    assert out_path.exists()

