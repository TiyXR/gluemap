import torch

def intrinsics_averaging(
    intrinsics_all, communities, intrinsics_mapping, camera_model="PINHOLE"
):
    max_idx = max(intrinsics_mapping.values())
    intrinsics = [[] for _ in range(max_idx + 1)]

    for idx_cluster, community in enumerate(communities):
        for i, idx in enumerate(community):
            intrinsics[intrinsics_mapping[idx]].append(
                intrinsics_all[idx_cluster][:, i]
            )

    # Now, we need to average the intrinsics for each type of camera
    for i in range(len(intrinsics)):
        if len(intrinsics[i]) == 0:
            intrinsics[i] = None
            continue

        intrinsics_curr = torch.stack(intrinsics[i])
        if camera_model.startswith("SIMPLE"):
            focals = (intrinsics_curr[:, :, 0, 0] + intrinsics_curr[:, :, 1, 1]) / 2
            intrinsics_curr[:, :, 0, 0] = focals
            intrinsics_curr[:, :, 1, 1] = focals

        intrinsics[i] = torch.median(intrinsics_curr, dim=0).values

    return intrinsics
