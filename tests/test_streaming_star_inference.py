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
                "indexes": torch.tensor([center, 1 - center]),
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


class FakeTokenCacheLocalInference:
    def encoder_token_cache_report(self):
        return {
            "contractId": "jarailsense.pi3-encoder-token-cache/v1",
            "capacityFrames": 4,
            "requestedFrameCount": 4,
            "cacheHitCount": 2,
            "cacheMissCount": 2,
            "encodedFrameCount": 2,
            "evictionCount": 0,
            "residentFrameCount": 2,
            "peakResidentFrameCount": 2,
            "peakResidentLogicalBytes": 128,
            "hitRate": 0.5,
            "preprocessingIdentity": "pi3-imagenet-normalize/v1",
            "dtype": "torch.bfloat16",
            "device": "cpu",
        }


class FakeTokenCacheBatchInference(FakeBatchInference):
    local_inference = FakeTokenCacheLocalInference()


class FakePipeline(StreamingStarInferencePipeline):
    def _load_models(self):
        self.models = {
            "pi3": torch.nn.Identity(),
            "vggsfm": torch.nn.Identity(),
        }
        return self.models

    def _create_batch_inference(self, models):
        return FakeBatchInference()


class FakeTokenCachePipeline(FakePipeline):
    def _create_batch_inference(self, models):
        return FakeTokenCacheBatchInference()


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
    assert report["profileContractId"] == (
        "jarailsense.gluemap-streaming-frontend-profile/v1"
    )
    assert report["modelLoadCount"] == 1
    assert report["pi3EncoderUniqueFrameCount"] == 2
    assert report["pi3EncoderInvocationFrameCount"] == 4
    assert report["duplicateEncoderFrameInvocationCount"] == 2
    assert report["hotPathEmptyCacheCount"] == 0
    assert report["timingSummary"]["forward"]["count"] == 2
    assert sum(value["starCount"] for value in report["cudaTimeline"]) == 2


def test_streaming_profile_uses_actual_token_cache_encoder_count():
    pipeline = FakeTokenCachePipeline(
        args(), 1, 0, file_name="unused.pth", device="cpu"
    )
    report = pipeline.run_to_sink(
        FakeDataset(), lambda _center, _output: None
    )
    assert report["encoderAccountingMode"] == "frame-token-cache/v1"
    assert report["pi3EncoderUniqueFrameCount"] == 2
    assert report["pi3EncoderInvocationFrameCount"] == 2
    assert report["duplicateEncoderFrameInvocationCount"] == 0
    assert report["encoderTokenCache"]["hitRate"] == 0.5


def test_timing_summary_reports_linear_percentiles():
    from gluemap.controllers.streaming_star_inference import timing_summary

    summary = timing_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["totalSeconds"] == 10.0
    assert summary["p50Seconds"] == 2.5
    assert summary["p95Seconds"] == 3.85


def test_streaming_profile_reads_live_cuda_counters():
    if not torch.cuda.is_available():
        return
    pipeline = FakePipeline(
        args(), 1, 0, file_name="unused.pth", device="cuda:0"
    )
    report = pipeline.run_to_sink(
        FakeDataset(), lambda _center, _output: None
    )
    assert report["device"] == "cuda:0"
    assert report["streamSynchronizationCount"] == 4
    assert report["synchronizationCount"] == 4
    assert report["peakCudaAllocatedBytes"] >= 0
    assert report["peakCudaReservedBytes"] >= 0
    assert len(report["cudaTimeline"]) >= 1


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
