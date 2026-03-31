"""
SIM(3) loop closure optimizer for GlueMap mini-stars.

Ported from VGGT-Long/loop_utils/sim3loop.py and
VGGT-Long/fastloop/solve_python.py.

Typical usage:
    sim3_dict = align_all_edges(worldpoints_dict, predictions_dict)
    num_stars = len(predictions_dict["indexes"])
    sequential_transforms = build_sequential_transforms(sim3_dict, num_stars)
    loop_constraints = [...]   # (i, j, (s, R, t)) non-adjacent pairs
    optimizer = Sim3LoopOptimizer()
    optimized = optimizer.optimize(sequential_transforms, loop_constraints)
"""

import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

try:
    import pypose as pp
    _PYPOSE_AVAILABLE = True
except ImportError:
    _PYPOSE_AVAILABLE = False
    print("PyPose not available — Sim3LoopOptimizer will not function.")

try:
    import sim3solve
    _CPP_SOLVER_AVAILABLE = True
except ImportError:
    _CPP_SOLVER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sparse linear solver (from VGGT-Long/fastloop/solve_python.py)
# ---------------------------------------------------------------------------

def _solve_sparse(A, b, freen: int):
    if freen < 0:
        return spsolve(A, b)
    else:
        A_sub = A[:freen, :freen].tocsc()
        b_sub = b[:freen]
        delta_sub = spsolve(A_sub, b_sub)
        delta = np.zeros_like(b)
        delta[:freen] = delta_sub
        return delta


def _solve_system_py(J_Ginv_i, J_Ginv_j, ii, jj, res, ep, lm, freen):
    device = res.device
    J_Ginv_i = J_Ginv_i.cpu()
    J_Ginv_j = J_Ginv_j.cpu()
    ii = ii.cpu()
    jj = jj.cpu()
    res = res.clone().cpu()

    r = res.size(0)
    n = max(ii.max().item(), jj.max().item()) + 1

    res_vec = res.view(-1).numpy().astype(np.float64)

    rows, cols, data = [], [], []
    ii_np = ii.numpy()
    jj_np = jj.numpy()
    J_Ginv_i_np = J_Ginv_i.numpy()
    J_Ginv_j_np = J_Ginv_j.numpy()

    for x in range(r):
        i = ii_np[x]
        j = jj_np[x]
        if i == j:
            raise ValueError("Self-edges are not allowed")
        for k in range(7):
            for l in range(7):
                row_idx = x * 7 + k
                col_idx_i = i * 7 + l
                rows.append(row_idx); cols.append(col_idx_i); data.append(J_Ginv_i_np[x, k, l])
                col_idx_j = j * 7 + l
                rows.append(row_idx); cols.append(col_idx_j); data.append(J_Ginv_j_np[x, k, l])

    J = coo_matrix((data, (rows, cols)), shape=(r * 7, n * 7)).tocsc()
    b_vec = -J.T @ res_vec
    A_mat = J.T @ J
    diag = A_mat.diagonal()
    A_mat.setdiag(diag * (1.0 + lm) + ep)

    freen_total = freen * 7
    delta = _solve_sparse(A_mat.tocsc(), b_vec, freen_total)
    return torch.from_numpy(delta.astype(np.float32)).view(n, 7).to(device)


# ---------------------------------------------------------------------------
# Helper: build sequential transforms from sim3_dict
# ---------------------------------------------------------------------------

def build_sequential_transforms(
    sim3_dict: Dict[Tuple[int, int], Tuple[float, np.ndarray, np.ndarray]],
    num_stars: int,
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """Build sequential (adjacent) SIM(3) transforms from sim3_dict.

    For sequential datasets stars are indexed 0, 1, …, N-1 in
    predictions_dict["indexes"] order. We extract sim3_dict[(k, k+1)]
    for k in range(num_stars - 1). Missing edges default to identity.

    Args:
        sim3_dict:  pairwise SIM(3) from align_all_edges()
        num_stars:  total number of mini-stars

    Returns:
        List of (s, R, t) with length num_stars - 1.
    """
    sequential = []
    for k in range(num_stars - 1):
        if (k, k + 1) in sim3_dict:
            sequential.append(sim3_dict[(k, k + 1)])
        else:
            sequential.append((1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)))
    return sequential


# ---------------------------------------------------------------------------
# Sim3LoopOptimizer (ported from VGGT-Long/loop_utils/sim3loop.py)
# ---------------------------------------------------------------------------

class Sim3LoopOptimizer:
    """
    Loop closure optimizer for sequences of SIM(3) transformations.

    Input:
      sequential_transforms: List[(s, R(3,3), t(3,))]
          Adjacent relative transforms between consecutive stars.
      loop_constraints: List[(i, j, (s, R, t))]
          Non-adjacent loop closure constraints.

    Output:
      Optimized sequential_transforms (same format).
    """

    def __init__(self, device: str = "cpu", solve_system_version: str = "python"):
        self.device = device
        if not _CPP_SOLVER_AVAILABLE and solve_system_version == "cpp":
            print("sim3solve C++ not available, falling back to Python solver.")
            solve_system_version = "python"
        self.solve_system_version = solve_system_version

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def numpy_to_pypose_sim3(self, s: float, R_mat: np.ndarray, t_vec: np.ndarray):
        """Convert (s, R, t) to pypose Sim3. Format: [t_x, t_y, t_z, q_x, q_y, q_z, q_w, s]"""
        q = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]
        data = np.concatenate([t_vec, q, np.array([s])])
        return pp.Sim3(torch.from_numpy(data).float().to(self.device))

    def pypose_sim3_to_numpy(self, sim3) -> Tuple[float, np.ndarray, np.ndarray]:
        """Convert pypose Sim3 to (s, R(3,3), t(3,))."""
        data = sim3.data.cpu().numpy()
        t = data[:3]
        q = data[3:7]  # [x, y, z, w]
        s = data[7]
        R_mat = R.from_quat(q).as_matrix()
        return s, R_mat, t

    # ------------------------------------------------------------------
    # Pose chain helpers
    # ------------------------------------------------------------------

    def sequential_to_absolute_poses(
        self, sequential_transforms: List[Tuple[float, np.ndarray, np.ndarray]]
    ) -> torch.Tensor:
        """Chain relative transforms into absolute poses starting from identity.

        S_01, S_12, … → T_0, T_1, T_2, …
        where T_i is the transform from world coordinates to frame i.
        """
        identity = pp.Sim3(torch.tensor([0., 0., 0., 0., 0., 0., 1., 1.], device=self.device))
        poses = [identity]
        current_pose = identity
        for s, R_mat, t_vec in sequential_transforms:
            rel = self.numpy_to_pypose_sim3(s, R_mat, t_vec)
            current_pose = current_pose @ rel
            poses.append(current_pose)
        return torch.stack(poses)

    def absolute_to_sequential_transforms(
        self, absolute_poses
    ) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """Inverse of sequential_to_absolute_poses.

        T_0, T_1, … → S_01, S_12, …
        """
        sequential = []
        n = absolute_poses.shape[0]
        for i in range(n - 1):
            rel = absolute_poses[i].Inv() @ absolute_poses[i + 1]
            sequential.append(self.pypose_sim3_to_numpy(rel))
        return sequential

    # ------------------------------------------------------------------
    # Loop constraint builder
    # ------------------------------------------------------------------

    def build_loop_constraints(
        self,
        loop_constraints: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray]]],
    ):
        if not loop_constraints:
            return (
                pp.Sim3(torch.empty(0, 8, device=self.device)),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

        loop_transforms, ii_loop, jj_loop = [], [], []
        for i, j, (s, R_mat, t_vec) in loop_constraints:
            loop_sim3 = self.numpy_to_pypose_sim3(s, R_mat, t_vec)
            loop_transforms.append(loop_sim3.data)
            ii_loop.append(i)
            jj_loop.append(j)

        dSloop = pp.Sim3(torch.stack(loop_transforms))
        ii_loop = torch.tensor(ii_loop, dtype=torch.long, device=self.device)
        jj_loop = torch.tensor(jj_loop, dtype=torch.long, device=self.device)
        return dSloop, ii_loop, jj_loop

    # ------------------------------------------------------------------
    # Residual computation
    # ------------------------------------------------------------------

    def residual(self, Ginv, input_poses, dSloop, ii, jj, jacobian=False):
        def _residual(C, Gi, Gj):
            out = C @ pp.Exp(Gi) @ pp.Exp(Gj).Inv()
            return out.Log().tensor()

        pred_inv_poses = pp.Sim3(input_poses).Inv()

        n, _ = pred_inv_poses.shape
        if n > 1:
            kk = torch.arange(1, n, device=self.device)
            ll = kk - 1
            Ti = pred_inv_poses[kk]
            Tj = pred_inv_poses[ll]
            dSij = Tj @ Ti.Inv()
        else:
            kk = torch.empty(0, dtype=torch.long, device=self.device)
            ll = torch.empty(0, dtype=torch.long, device=self.device)
            dSij = pp.Sim3(torch.empty(0, 8, device=self.device))

        constants = (
            torch.cat((dSij.data, dSloop.data), dim=0)
            if dSloop.shape[0] > 0
            else dSij.data
        )
        if constants.shape[0] > 0:
            constants = pp.Sim3(constants)
            iii = torch.cat((kk, ii))
            jjj = torch.cat((ll, jj))
            resid = _residual(constants, Ginv[iii], Ginv[jjj])
        else:
            iii = torch.empty(0, dtype=torch.long, device=self.device)
            jjj = torch.empty(0, dtype=torch.long, device=self.device)
            resid = torch.empty(0, device=self.device)

        if not jacobian:
            return resid

        if constants.shape[0] > 0:
            def batch_jacobian(func, x):
                def _func_sum(*x):
                    return func(*x).sum(dim=0)
                _, b, c = torch.autograd.functional.jacobian(_func_sum, x, vectorize=True)
                from einops import rearrange
                return rearrange(torch.stack((b, c)), "N O B I -> N B O I", N=2)

            J_Ginv_i, J_Ginv_j = batch_jacobian(_residual, (constants, Ginv[iii], Ginv[jjj]))
        else:
            J_Ginv_i = torch.empty(0, device=self.device)
            J_Ginv_j = torch.empty(0, device=self.device)

        return resid, (J_Ginv_i, J_Ginv_j, iii, jjj)

    # ------------------------------------------------------------------
    # Main optimization
    # ------------------------------------------------------------------

    def optimize(
        self,
        sequential_transforms: List[Tuple[float, np.ndarray, np.ndarray]],
        loop_constraints: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray]]],
        max_iterations: int = 100,
        lambda_init: float = 1e-4,
    ) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Jointly optimize SIM(3) absolute poses given sequential + loop constraints.

        Args:
            sequential_transforms: n-1 relative transforms between consecutive stars.
            loop_constraints:      Non-adjacent closure constraints (i, j, (s, R, t)).
            max_iterations:        Maximum L-M iterations.
            lambda_init:           Initial damping factor for L-M.

        Returns:
            Optimized sequential_transforms (same format as input).
        """
        if not loop_constraints:
            print("Warning: No loop constraints provided, returning original transforms.")
            return sequential_transforms

        input_poses = self.sequential_to_absolute_poses(sequential_transforms)
        dSloop, ii_loop, jj_loop = self.build_loop_constraints(loop_constraints)

        Ginv = pp.Sim3(input_poses).Inv().Log()
        lmbda = lambda_init
        residual_history = []

        print(
            f"Starting Sim3 loop optimization: {len(sequential_transforms) + 1} poses, "
            f"{len(loop_constraints)} loop constraints"
        )

        for itr in range(max_iterations):
            resid, (J_Ginv_i, J_Ginv_j, iii, jjj) = self.residual(
                Ginv, input_poses, dSloop, ii_loop, jj_loop, jacobian=True
            )

            if resid.numel() == 0:
                print("No residuals to optimize.")
                break

            current_cost = resid.square().mean().item()
            residual_history.append(current_cost)

            try:
                t0 = time.time()
                if self.solve_system_version == "cpp":
                    (delta_pose,) = sim3solve.solve_system(
                        J_Ginv_i, J_Ginv_j, iii, jjj, resid, 0.0, lmbda, -1
                    )
                else:
                    delta_pose = _solve_system_py(
                        J_Ginv_i, J_Ginv_j, iii, jjj, resid, 0.0, lmbda, -1
                    )
                t1 = time.time()
            except Exception as e:
                print(f"Solver failed at iteration {itr}: {e}")
                break

            Ginv_tmp = Ginv + delta_pose
            new_resid = self.residual(Ginv_tmp, input_poses, dSloop, ii_loop, jj_loop)
            new_cost = (
                new_resid.square().mean().item() if new_resid.numel() > 0 else float("inf")
            )

            if new_cost < current_cost:
                Ginv = Ginv_tmp
                lmbda /= 2
                status = "accepted"
            else:
                lmbda *= 2
                status = "rejected"

            print(
                f"Iter {itr:3d}: cost {current_cost:.8f} -> {new_cost:.8f} ({status}) | "
                f"solver ({self.solve_system_version}): {(t1 - t0) * 1000:.2f} ms"
            )

            if current_cost < 1e-5 and itr >= 4 and len(residual_history) >= 5:
                if residual_history[-5] / residual_history[-1] < 1.5:
                    print(f"Converged at iteration {itr}.")
                    break

        optimized_absolute = pp.Exp(Ginv).Inv()
        optimized_sequential = self.absolute_to_sequential_transforms(optimized_absolute)

        final_cost = residual_history[-1] if residual_history else float("nan")
        print(f"Sim3 loop optimization done. Final cost: {final_cost:.8f}")

        return optimized_sequential
