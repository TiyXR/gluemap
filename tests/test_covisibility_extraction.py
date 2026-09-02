from __future__ import annotations

import networkx as nx
import torch

from gluemap.estimators.covisibility_extraction import CovisibilityExtraction


def _virtual_track_inputs() -> tuple[torch.Tensor, ...]:
    depth = torch.linspace(1.0, 4.0, 28 * 28).reshape(1, 1, 28, 28, 1)
    depth = depth.expand(-1, 3, -1, -1, -1).clone()
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4).expand(1, 3, -1, -1).clone()
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 3, -1, -1).clone()
    valid_mask = torch.ones((1, 3, 28, 28), dtype=torch.bool)
    indexes = torch.tensor([[10, 11, 12]])
    return depth, extrinsics, intrinsics, valid_mask, indexes


def test_virtual_track_noise_is_reproducible_for_one_star() -> None:
    inputs = _virtual_track_inputs()
    extractor = CovisibilityExtraction(
        virtual_track_depth_noise_ratio=0.1,
        virtual_track_noise_seed=20260831,
    )

    first = extractor._calculate_virtual_tracks(*inputs)
    second = extractor._calculate_virtual_tracks(*inputs)

    for first_value, second_value in zip(first, second, strict=True):
        assert torch.equal(first_value, second_value)


def test_virtual_track_noise_changes_with_star_identity() -> None:
    depth, extrinsics, intrinsics, valid_mask, indexes = _virtual_track_inputs()
    extractor = CovisibilityExtraction(
        virtual_track_depth_noise_ratio=0.1,
        virtual_track_noise_seed=20260831,
    )

    first = extractor._calculate_virtual_tracks(
        depth, extrinsics, intrinsics, valid_mask, indexes
    )
    second = extractor._calculate_virtual_tracks(
        depth, extrinsics, intrinsics, valid_mask, indexes + 1
    )

    assert not torch.equal(first[1], second[1])


def test_zero_virtual_track_noise_does_not_consume_random_state() -> None:
    inputs = _virtual_track_inputs()
    extractor = CovisibilityExtraction(virtual_track_depth_noise_ratio=0.0)
    torch.manual_seed(71)
    expected = torch.rand(4)
    torch.manual_seed(71)

    extractor._calculate_virtual_tracks(*inputs)

    assert torch.equal(torch.rand(4), expected)


def test_main_returns_validity_not_negative_mask(monkeypatch) -> None:
    extractor = CovisibilityExtraction(include_track=False, return_cpu=False)
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3)
    depth = torch.ones((1, 1, 2, 2, 1))
    indexes = torch.tensor([[7]])
    scores = torch.ones((1, 1))
    pairwise_valid = torch.ones((1, 1, 2, 2), dtype=torch.bool)
    tracks_virtual = torch.zeros((1, 1, 4, 2))
    points3d_virtual = torch.zeros((1, 4, 3))
    valid_virtual = torch.ones((1, 1, 4), dtype=torch.bool)
    isnegative_virtual = torch.zeros((1, 1, 4), dtype=torch.bool)

    monkeypatch.setattr(
        extractor,
        "_convert_from_depth_to_world_points",
        lambda *_args: torch.zeros((1, 1, 2, 2, 3)),
    )
    monkeypatch.setattr(
        extractor,
        "_verify_by_reprojection_n2",
        lambda *_args, **_kwargs: (scores, pairwise_valid),
    )
    monkeypatch.setattr(
        extractor,
        "_calculate_virtual_tracks",
        lambda *_args: (
            tracks_virtual,
            points3d_virtual,
            valid_virtual,
            isnegative_virtual,
        ),
    )

    outputs = extractor.main(
        {"extrinsics": extrinsics, "intrinsics": intrinsics, "depth": depth},
        indexes,
        torch.zeros((1, 1, 4)),
    )

    assert torch.equal(outputs[-1], valid_virtual)
    assert not torch.equal(outputs[-1], isnegative_virtual)


def test_planned_star_preserves_indirect_dense_path(monkeypatch) -> None:
    extractor = CovisibilityExtraction(graph_policy="planned-star")
    calls = []
    pair_scores = torch.tensor(
        [
            [1.0, 0.5, 0.1],
            [0.5, 1.0, 0.5],
            [0.1, 0.5, 1.0],
        ]
    )

    def verify(world_points, *_args, **kwargs):
        view_ids = world_points[0, :, 0, 0, 0].to(dtype=torch.long)
        calls.append((view_ids.tolist(), kwargs))
        scores = pair_scores[view_ids[0], view_ids].unsqueeze(0)
        valid = torch.ones((1, len(view_ids), 2, 2), dtype=torch.bool)
        return scores, valid

    monkeypatch.setattr(extractor, "_verify_by_reprojection", verify)
    world = torch.zeros((1, 3, 2, 2, 3))
    world[:, 1, :, :, 0] = 1
    world[:, 2, :, :, 0] = 2
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4).expand(1, 3, -1, -1)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 3, -1, -1)

    actual_scores, actual_valid = extractor._verify_by_reprojection_star(
        world, extrinsics, intrinsics
    )

    assert torch.equal(actual_scores, torch.tensor([[1.0, 0.5, 0.25]]))
    assert torch.equal(actual_valid, torch.ones((1, 3, 2, 2), dtype=torch.bool))
    assert [call[0] for call in calls] == [[0, 1, 2], [1, 2]]
    assert all(call[1]["symmetric_scores"] is True for call in calls)


def test_torch_transitive_scores_match_networkx_reference() -> None:
    scores = torch.tensor(
        [
            [
                [1.0, 0.73, 0.0, 0.18],
                [0.73, 1.0, 0.61, 0.52],
                [0.0, 0.61, 1.0, 0.88],
                [0.18, 0.52, 0.88, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.4, 0.0],
                [0.0, 0.4, 1.0, 0.3],
                [0.0, 0.0, 0.3, 1.0],
            ],
        ],
        dtype=torch.float32,
    )
    expected_rows = []
    for batch_scores in scores:
        valid_i, valid_j = torch.where(batch_scores > 0.0)
        edges = {
            (
                int(valid_i[index]),
                int(valid_j[index]),
                -torch.log(batch_scores[valid_i[index], valid_j[index]]).item(),
            )
            for index in range(len(valid_i))
        }
        graph = nx.Graph()
        graph.add_weighted_edges_from(edges)
        lengths = nx.single_source_dijkstra_path_length(graph, 0, weight="weight")
        expected_rows.append(
            torch.exp(
                -torch.tensor(
                    [lengths.get(index, float("inf")) for index in range(4)]
                )
            )
        )

    actual = CovisibilityExtraction._aggregate_transitive_scores_torch(scores)

    assert torch.equal(actual, torch.stack(expected_rows))


def test_planned_star_matches_dense_reprojection_on_real_tensors() -> None:
    depth = torch.linspace(3.0, 5.0, 8 * 10).reshape(1, 1, 8, 10, 1)
    depth = depth.expand(-1, 4, -1, -1, -1).clone()
    depth[:, 1] += 0.05
    depth[:, 2] -= 0.03
    depth[:, 3] += 0.08
    extrinsics = (
        torch.eye(4).reshape(1, 1, 4, 4).expand(1, 4, -1, -1).clone()
    )
    extrinsics[:, 1, 0, 3] = 0.04
    extrinsics[:, 2, 1, 3] = -0.03
    extrinsics[:, 3, 0, 3] = 0.07
    intrinsics = (
        torch.tensor(
            [[6.0, 0.0, 4.5], [0.0, 6.0, 3.5], [0.0, 0.0, 1.0]]
        )
        .reshape(1, 1, 3, 3)
        .expand(1, 4, -1, -1)
        .clone()
    )
    extractor = CovisibilityExtraction(return_cpu=False)
    world_points = extractor._convert_from_depth_to_world_points(
        depth, extrinsics, intrinsics
    )

    dense_scores, dense_valid = extractor._verify_by_reprojection_n2(
        world_points, extrinsics, intrinsics
    )
    planned_scores, planned_valid = extractor._verify_by_reprojection_star(
        world_points, extrinsics, intrinsics
    )

    assert torch.equal(planned_scores, dense_scores)
    assert torch.equal(planned_valid, dense_valid)


def test_invalid_covisibility_parameters_are_rejected() -> None:
    for kwargs in (
        {"graph_policy": "complete"},
        {"virtual_track_depth_noise_ratio": -0.1},
        {"virtual_track_noise_seed": -1},
    ):
        try:
            CovisibilityExtraction(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid parameters were accepted: {kwargs}")
