import pytest
import torch
import numpy as np

from gluemap.estimators.fixed_lag_prior import (
    FejPosePriorCostFunction,
    FejPriorState,
    FixedLagPriorError,
    marginalize_linearized_tracks,
    marginalize_ceres_linearization,
)
from gluemap.estimators.fixed_lag_ceres_linearization import (
    CeresProblemLinearization,
)


def _fixture(dtype=torch.float64):
    generator = torch.Generator().manual_seed(9127)
    camera_ids = (10, 11, 12)
    point_count = 8
    maximum_views = 3
    camera_indexes = torch.tensor(
        [[0, 1, 2], [0, 1, -1], [1, 2, -1], [0, 2, -1]] * 2,
        dtype=torch.int64,
    )
    residuals = torch.randn(
        point_count, maximum_views, 2, generator=generator, dtype=dtype
    )
    camera_jacobians = torch.randn(
        point_count, maximum_views, 2, 6, generator=generator, dtype=dtype
    )
    point_jacobians = torch.randn(
        point_count, maximum_views, 2, 3, generator=generator, dtype=dtype
    )
    mask = camera_indexes >= 0
    residuals[~mask] = 0
    camera_jacobians[~mask] = 0
    point_jacobians[~mask] = 0
    linearization = torch.randn(3, 7, generator=generator, dtype=dtype)
    linearization[:, :4] /= torch.linalg.vector_norm(
        linearization[:, :4], dim=1, keepdim=True
    )
    initial_hessian = torch.eye(18, dtype=dtype) * 0.25
    initial_gradient = torch.randn(18, generator=generator, dtype=dtype) * 0.01
    eigenvalues, eigenvectors = torch.linalg.eigh(initial_hessian)
    factor = eigenvalues.sqrt()[:, None] * eigenvectors.T
    factor_residual = (eigenvectors.T @ initial_gradient) / eigenvalues.sqrt()
    previous = FejPriorState(
        camera_ids=camera_ids,
        linearization_points=linearization.clone(),
        hessian=initial_hessian,
        gradient=initial_gradient,
        factor=factor,
        factor_residual=factor_residual,
        report={},
    )
    return {
        "camera_ids": camera_ids,
        "linearization_points": linearization,
        "observation_camera_indexes": camera_indexes,
        "residuals": residuals,
        "camera_jacobians": camera_jacobians,
        "point_jacobians": point_jacobians,
        "eliminate_camera_id": 10,
        "previous_prior": previous,
        "relative_rank_threshold": 1e-12,
    }


def _dense_reference(values):
    camera_ids = values["camera_ids"]
    camera_indexes = values["observation_camera_indexes"]
    residuals = values["residuals"]
    camera_jacobians = values["camera_jacobians"]
    point_jacobians = values["point_jacobians"]
    point_count, maximum_views = camera_indexes.shape
    camera_dimension = len(camera_ids) * 6
    point_dimension = point_count * 3
    rows = int((camera_indexes >= 0).sum().item()) * 2
    jacobian = torch.zeros(
        rows, camera_dimension + point_dimension, dtype=torch.float64
    )
    residual = torch.zeros(rows, dtype=torch.float64)
    row = 0
    for point in range(point_count):
        for view in range(maximum_views):
            camera = int(camera_indexes[point, view])
            if camera < 0:
                continue
            jacobian[row : row + 2, camera * 6 : (camera + 1) * 6] = (
                camera_jacobians[point, view]
            )
            jacobian[
                row : row + 2,
                camera_dimension + point * 3 : camera_dimension + (point + 1) * 3,
            ] = point_jacobians[point, view]
            residual[row : row + 2] = residuals[point, view]
            row += 2
    hessian = jacobian.T @ jacobian
    gradient = jacobian.T @ residual
    previous = values["previous_prior"]
    hessian[:camera_dimension, :camera_dimension] += previous.hessian
    gradient[:camera_dimension] += previous.gradient
    point_columns = torch.arange(camera_dimension, camera_dimension + point_dimension)
    camera_columns = torch.arange(camera_dimension)
    h_pp = hessian[point_columns[:, None], point_columns]
    h_pc = hessian[point_columns[:, None], camera_columns]
    h_cc = hessian[camera_columns[:, None], camera_columns]
    g_p = gradient[point_columns]
    g_c = gradient[camera_columns]
    point_inverse = torch.linalg.pinv(h_pp, rtol=1e-12)
    pose_hessian = h_cc - h_pc.T @ point_inverse @ h_pc
    pose_gradient = g_c - h_pc.T @ point_inverse @ g_p
    marginal = torch.arange(0, 6)
    retained = torch.arange(6, camera_dimension)
    h_mm = pose_hessian[marginal[:, None], marginal]
    h_mr = pose_hessian[marginal[:, None], retained]
    h_rr = pose_hessian[retained[:, None], retained]
    g_m = pose_gradient[marginal]
    g_r = pose_gradient[retained]
    inverse_mm = torch.linalg.pinv(h_mm, rtol=1e-12)
    return (
        h_rr - h_mr.T @ inverse_mm @ h_mr,
        g_r - h_mr.T @ inverse_mm @ g_m,
    )


def test_batched_schur_matches_dense_reference_on_cpu():
    values = _fixture()
    expected_hessian, expected_gradient = _dense_reference(values)

    result = marginalize_linearized_tracks(**values, device_policy="cpu")

    assert result.camera_ids == (11, 12)
    assert result.report["status"] == "passed"
    assert result.report["gpuUsed"] is False
    torch.testing.assert_close(result.hessian, expected_hessian, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(result.gradient, expected_gradient, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        result.factor.T @ result.factor,
        result.hessian,
        rtol=1e-9,
        atol=1e-9,
    )
    torch.testing.assert_close(
        result.factor.T @ result.factor_residual,
        result.gradient,
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_schur_matches_cpu_and_reports_real_gpu_use():
    values = _fixture()
    cpu = marginalize_linearized_tracks(**values, device_policy="cpu")

    cuda = marginalize_linearized_tracks(
        **values, device_policy="cuda-required"
    ).cpu()

    assert cuda.report["gpuUsed"] is True
    assert cuda.report["backend"] == "cuda"
    torch.testing.assert_close(cuda.hessian, cpu.hessian, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(cuda.gradient, cpu.gradient, rtol=1e-8, atol=1e-8)


def test_previous_fej_linearization_identity_cannot_change():
    values = _fixture()
    values["linearization_points"] = values["linearization_points"].clone()
    values["linearization_points"][0, 0] += 1e-12

    with pytest.raises(FixedLagPriorError, match="linearization point changed"):
        marginalize_linearized_tracks(**values, device_policy="cpu")


def test_track_requires_two_real_views():
    values = _fixture()
    values["observation_camera_indexes"] = values[
        "observation_camera_indexes"
    ].clone()
    values["observation_camera_indexes"][0, 1:] = -1

    with pytest.raises(FixedLagPriorError, match="fewer than two views"):
        marginalize_linearized_tracks(**values, device_policy="cpu")


def test_dense_fej_cost_moves_one_pose_on_ceres_manifold():
    import numpy as np
    import pyceres
    import pygluemap

    linearization = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [0.02, -0.01, 0.03, 1.0, -2.0, 0.5], dtype=torch.float64
    )
    prior = FejPriorState(
        camera_ids=(7,),
        linearization_points=linearization,
        hessian=torch.eye(6, dtype=torch.float64),
        gradient=-target,
        factor=torch.eye(6, dtype=torch.float64),
        factor_residual=-target,
        report={},
    )
    pose = linearization[0].numpy().copy()
    problem = pyceres.Problem()
    problem.add_parameter_block(pose, 7, pygluemap.CreatePoseManifold())
    cost = FejPosePriorCostFunction(prior)
    problem.add_residual_block(cost, None, [pose])
    options = pyceres.SolverOptions()
    options.max_num_iterations = 20
    summary = pyceres.SolverSummary()

    pyceres.solve(options, problem, summary)

    quaternion = pose[:4]
    recovered_rotation = quaternion[:3] / np.linalg.norm(quaternion[:3])
    recovered_rotation *= np.arctan2(
        np.linalg.norm(quaternion[:3]), quaternion[3]
    )
    np.testing.assert_allclose(recovered_rotation, target[:3].numpy(), atol=1e-9)
    np.testing.assert_allclose(pose[4:], target[3:].numpy(), atol=1e-9)


def test_ceres_crs_path_matches_direct_dense_elimination():
    values = _fixture()
    camera_indexes = values["observation_camera_indexes"]
    residuals = values["residuals"]
    camera_jacobians = values["camera_jacobians"]
    point_jacobians = values["point_jacobians"]
    point_count, maximum_views = camera_indexes.shape
    camera_dimension = len(values["camera_ids"]) * 6
    point_dimension = point_count * 3
    rows = int((camera_indexes >= 0).sum().item()) * 2
    jacobian = torch.zeros(
        rows, camera_dimension + point_dimension, dtype=torch.float64
    )
    residual = torch.zeros(rows, dtype=torch.float64)
    row = 0
    for point in range(point_count):
        for view in range(maximum_views):
            camera = int(camera_indexes[point, view])
            if camera < 0:
                continue
            jacobian[row : row + 2, camera * 6 : (camera + 1) * 6] = (
                camera_jacobians[point, view]
            )
            jacobian[
                row : row + 2,
                camera_dimension + point * 3 : camera_dimension + (point + 1) * 3,
            ] = point_jacobians[point, view]
            residual[row : row + 2] = residuals[point, view]
            row += 2
    previous = values["previous_prior"]
    prior_rows = previous.factor.shape[0]
    prior_jacobian = torch.zeros(
        prior_rows, camera_dimension + point_dimension, dtype=torch.float64
    )
    prior_jacobian[:, :camera_dimension] = previous.factor
    jacobian = torch.cat((jacobian, prior_jacobian), dim=0)
    residual = torch.cat((residual, previous.factor_residual), dim=0)
    sparse = jacobian.to_sparse_csr()
    linearization = CeresProblemLinearization(
        camera_ids=values["camera_ids"],
        image_ids=(1, 2, 3),
        point3d_ids=tuple(range(point_count)),
        pose_ambient_values=values["linearization_points"].numpy(),
        point_values=np.zeros((point_count, 3), dtype=np.float64),
        residuals=residual.numpy(),
        row_offsets=sparse.crow_indices().numpy(),
        column_indices=sparse.col_indices().numpy(),
        jacobian_values=sparse.values().numpy(),
        report={
            "contractId": "jarailsense.gluemap-ceres-linearization/v1",
            "columnCount": camera_dimension + point_dimension,
        },
    )
    expected_hessian, expected_gradient = _dense_reference(values)

    result = marginalize_ceres_linearization(
        linearization,
        eliminate_camera_id=10,
        previous_prior=previous,
        device_policy="cpu",
        relative_rank_threshold=1e-12,
    )

    torch.testing.assert_close(result.hessian, expected_hessian, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(result.gradient, expected_gradient, rtol=1e-9, atol=1e-9)


def test_ceres_point_without_variable_camera_does_not_create_pose_constraint():
    residuals = np.zeros(14, dtype=np.float64)
    residuals[:2] = np.asarray([1.0, -0.5])
    jacobian = np.zeros((14, 15), dtype=np.float64)
    jacobian[:2, 12:] = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    jacobian[2:, :12] = np.eye(12, dtype=np.float64)
    sparse = torch.from_numpy(jacobian).to_sparse_csr()
    linearization = CeresProblemLinearization(
        camera_ids=(10, 11),
        image_ids=(1, 2),
        point3d_ids=(1,),
        pose_ambient_values=np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        point_values=np.zeros((1, 3), dtype=np.float64),
        residuals=residuals,
        row_offsets=sparse.crow_indices().numpy(),
        column_indices=sparse.col_indices().numpy(),
        jacobian_values=sparse.values().numpy(),
        report={
            "contractId": "jarailsense.gluemap-ceres-linearization/v1",
            "columnCount": 15,
        },
    )

    result = marginalize_ceres_linearization(
        linearization,
        eliminate_camera_id=10,
        device_policy="cpu",
    )

    assert result.camera_ids == (11,)
    assert result.report["pointWithoutVariableCameraCount"] == 1
    torch.testing.assert_close(result.hessian, torch.eye(6, dtype=torch.float64))
