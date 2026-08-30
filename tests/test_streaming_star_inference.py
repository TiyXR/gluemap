import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend
from torch.utils.data import IterableDataset

from gluemap.controllers.star_inference import (
    BatchInferenceStar,
    pi3_sdpa_compatibility,
    resolve_pi3_sdpa_backend,
)
from gluemap.controllers.streaming_star_inference import (
    StreamingStarInferenceError,
    StreamingStarInferencePipeline,
    move_output_to_cpu,
)


class FakeDataset(IterableDataset):
    gpu_resident_stream = True
    peak_resident_frames = 3
    released_frame_count = 2

    def __len__(self) -> int:
        return 2

    def __iter__(self):
        for center in range(2):
            yield {
                "star_indexes": center,
                "value": torch.tensor([float(center)]),
            }


class FakeBatchInference:
    def main(self, batch, use_dummy_tracks=False):
        center = int(batch["star_indexes"].item())
        return {
            "center": torch.tensor(center),
            "nested": [torch.tensor([center + 1])],
            "_forward_time": 0.25,
            "_track_time": 0.5,
        }


class FakePipeline(StreamingStarInferencePipeline):
    def _load_models(self):
        self.models = {
            "pi3": torch.nn.Identity(),
            "vggsfm": torch.nn.Identity(),
        }
        return self.models

    def _create_batch_inference(self, models):
        return FakeBatchInference()


def args(**overrides):
    values = {
        "batch_size": 1,
        "distributed": False,
        "use_dummy_tracks": False,
        "enforce_g0_clean_process": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_recursive_output_detach_keeps_structure():
    value = {"a": (torch.tensor([1]), [torch.tensor([2])])}
    moved = move_output_to_cpu(value)
    assert moved["a"][0].device.type == "cpu"
    assert moved["a"][1][0].tolist() == [2]


def test_star_output_binds_working_and_original_image_shapes():
    pipeline = BatchInferenceStar.__new__(BatchInferenceStar)
    predictions = {
        "track": torch.zeros((1, 2, 3, 2)),
        "vis": torch.ones((1, 2, 3)),
        "conf": torch.ones((1, 2, 3)),
    }
    pipeline._predict_images = lambda *_args, **_kwargs: (
        predictions,
        0.1,
        0.2,
    )

    class Covisibility:
        def main(self, *_args):
            return (
                torch.zeros((1, 2, 4, 4)),
                torch.zeros((1, 2, 3, 3)),
                torch.ones((1, 2)),
                torch.zeros((1, 2, 1, 2)),
                torch.zeros((1, 1, 3)),
                torch.ones((1, 2, 1)),
            )

    pipeline.covisibility_extraction = Covisibility()
    batch = {
        "indexes": torch.tensor([[4, 5]]),
        "images": torch.zeros((1, 2, 3, 294, 518)),
        "images_shape_ori": torch.tensor([[[1080, 1920], [1080, 1920]]]),
        "images_change": torch.tensor(
            [[[0.27, 0.27, 0.0, 0.0], [0.27, 0.27, 0.0, 0.0]]]
        ),
    }
    result = BatchInferenceStar.main(pipeline, batch)
    assert result["working_image_shape"] == [294, 518]
    assert result["images_shape_ori"] == [[1080, 1920], [1080, 1920]]
    assert len(result["images_change"]) == 2


def test_pre_ampere_pi3_attention_replaces_flash_with_math(monkeypatch):
    calls = []

    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda _device: (6, 1)
    )
    original = torch.nn.attention.sdpa_kernel

    def recording(backends):
        calls.append(backends)
        return original(SDPBackend.MATH)

    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", recording)
    assert resolve_pi3_sdpa_backend("cuda:0") == "math"
    with pi3_sdpa_compatibility("cuda:0") as backend:
        with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            pass
    assert backend == "math"
    assert calls == [[SDPBackend.MATH]]


def test_ampere_pi3_attention_keeps_native_flash(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda _device: (8, 9)
    )
    assert resolve_pi3_sdpa_backend("cuda:0") == "flash"
    with pi3_sdpa_compatibility("cuda:0") as backend:
        assert backend == "flash"


def test_streaming_pipeline_emits_contiguous_outputs_without_global_list():
    pipeline = FakePipeline(
        args(), 1, 0, file_name="unused.pth", device="cpu"
    )
    outputs = []
    report = pipeline.run_to_sink(
        FakeDataset(), lambda center, output: outputs.append((center, output))
    )

    assert [value[0] for value in outputs] == [0, 1]
    assert outputs[1][1]["nested"][0].tolist() == [2]
    assert report["starCount"] == 2
    assert report["loadedModelKeys"] == ["pi3", "vggsfm"]
    assert report["saladLoadCount"] == 0
    assert report["peakResidentFrames"] == 3
    assert report["releasedFrameCount"] == 2


def test_streaming_pipeline_rejects_distributed_execution():
    pipeline = FakePipeline(
        args(distributed=True), 2, 0, file_name="unused.pth", device="cpu"
    )
    try:
        pipeline.run_to_sink(FakeDataset(), lambda _center, _output: None)
    except StreamingStarInferenceError as error:
        assert "one process" in str(error)
    else:
        raise AssertionError("distributed streaming was not rejected")


def test_feature_extractor_import_does_not_eagerly_load_faiss():
    source_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys;"
        f"sys.path.insert(0, {str(source_root)!r});"
        "import gluemap.estimators.feature_extraction;"
        "assert not any(x == 'faiss' or x.startswith('faiss.') "
        "for x in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
