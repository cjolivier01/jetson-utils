import importlib


def test_cuda_memory_allocations_small():
    ju = importlib.import_module("jetson_utils")

    # Allocate raw memory (device)
    mem = ju.cudaMalloc(size=16)
    assert mem.size >= 16

    # Allocate mapped memory (zero-copy)
    mapped = ju.cudaAllocMapped(size=32)
    assert mapped.size >= 32
    assert mapped.mapped is True

    # Allocate small image buffer
    img = ju.cudaAllocMapped(width=4, height=4, format="rgb8")
    assert img.width == 4 and img.height == 4
    assert isinstance(img.timestamp, int)

