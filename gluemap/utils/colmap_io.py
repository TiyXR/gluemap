import pycolmap
import os
import numpy as np


from gluemap.utils.colmap import camera_from_intrinsics_matrix


def convert_to_colmap_format(
    images_shape_ori,
    predictions_dict,
    global_rotations,
    global_centers,
    global_tracks,
    global_intrinsics,
    intrinsics_mapping,
    indexes=None,
    images_list=None,
    pose_only=False,
    return_point3d_id_map=False,
    camera_type=None,
):
    reconstruction = pycolmap.Reconstruction()

    camera_sizes = [None for _ in range(len(global_intrinsics))]
    for i in range(len(intrinsics_mapping)):
        camera_id = intrinsics_mapping[i]
        if camera_sizes[camera_id] is None:
            camera_sizes[camera_id] = images_shape_ori[i]

    # Write cameras
    for i in range(len(global_intrinsics)):
        if global_intrinsics[i] is None:
            continue
        model = camera_type if camera_type is not None else "PINHOLE"
        camera = camera_from_intrinsics_matrix(
            global_intrinsics[i][0], model, camera_sizes[i][-1], camera_sizes[i][0], i + 1
        )
        reconstruction.add_camera_with_trivial_rig(camera)

    # Write images
    images_points2d = {
        i: [] for i in global_rotations.keys()
    }  # (image_id, [(point_id, point_2d), ...])
    # counter = 0
    points3d = []  # point_3d

    if indexes is None and not pose_only:
        indexes = range(len(predictions_dict["indexes"]))

    if not pose_only:
        max_track_length = max([predictions_dict["tracks"][idx].shape[2] for idx in indexes])
    else:
        max_track_length = 0

    points3d_idx = []
    if not pose_only:
        for idx in indexes:
            scores = predictions_dict["scores"][idx]
            for j in range(predictions_dict["tracks"][idx].shape[-2]):
                valid = scores[0, :, j] > 0.0
                if valid.sum() < 2:
                    continue

                points3d.append({"point3d": global_tracks[idx][j], "tracks": []})
                points3d_idx.append(idx * max_track_length + j)

                # for i, idx_inner in enumerate(predictions_dict["indexes"][idx]):
                #     if scores[0, i, j] <= 0:
                #         continue
                valid_idx = np.where(valid)[0].tolist()
                for i in valid_idx:
                    idx_inner = predictions_dict["indexes"][idx][i]
                    points3d[len(points3d) - 1]["tracks"].append(
                        (idx_inner, len(images_points2d[idx_inner]))
                    )
                    images_points2d[idx_inner].append(
                        (
                            len(points3d) - 1,
                            predictions_dict["tracks"][idx][0, i, j].cpu().numpy(),
                        )
                    )

                if points3d[len(points3d) - 1]["tracks"] == []:
                    points3d.pop()
                    continue

    # Add images
    for idx in images_points2d.keys():
        image = pycolmap.Image()
        image.camera_id = intrinsics_mapping[idx] + 1

        for point_id, point_2d in images_points2d[idx]:
            image.points2D.append(pycolmap.Point2D(point_2d))

        if images_list is not None:
            image.name = images_list[idx]
        else:
            image.name = str(idx)
        image.image_id = idx + 1

        cam_from_world = pycolmap.Rigid3d(
            np.concatenate(
                [global_rotations[idx][:3, :3].T, global_centers[idx].reshape(3, 1)],
                axis=1,
            )
        ).inverse()
        reconstruction.add_image_with_trivial_frame(image, cam_from_world)

    # Add 3D points
    point3d_id_map = {}
    for i, point_3d in enumerate(points3d):
        tracks = []
        for j, track in enumerate(point_3d["tracks"]):
            tracks.append(pycolmap.TrackElement(track[0] + 1, track[1]))  # 1-indexed image_id

            pass

        point3d_id = reconstruction.add_point3D(
            point_3d["point3d"].reshape((3, 1)), pycolmap.Track(tracks)
        )

        point3d_id_map[points3d_idx[i]] = point3d_id

    if return_point3d_id_map:
        return reconstruction, point3d_id_map
    else:
        return reconstruction


def write_to_colmap_format(
    dir_write,
    images_shape_ori,
    predictions_dict,
    global_rotations,
    global_centers,
    global_tracks,
    global_intrinsics,
    intrinsics_mapping,
    indexes=None,
    images_list=None,
    pose_only=False,
    camera_type=None,
):
    os.makedirs(dir_write, exist_ok=True)

    reconstruction = convert_to_colmap_format(
        images_shape_ori,
        predictions_dict,
        global_rotations,
        global_centers,
        global_tracks,
        global_intrinsics,
        intrinsics_mapping,
        indexes=indexes,
        images_list=images_list,
        pose_only=pose_only,
        camera_type=camera_type,
    )

    reconstruction.write(dir_write)
