import os
import subprocess
import sys


def test_video_viewer_runs_with_ss_and_t():
    # Path to the example script in runfiles
    script = os.path.join(os.path.dirname(__file__), "..", "examples", "video-viewer.py")
    script = os.path.normpath(script)
    assert os.path.exists(script), f"missing script: {script}"

    img_path = os.path.join("data", "fontmapA.png")
    assert os.path.exists(img_path)

    cmd = [sys.executable, script, f"file://{img_path}", "-ss", "0.0", "-t", "0.02", "--headless"]
    # Run with a short timeout—should exit quickly
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr

