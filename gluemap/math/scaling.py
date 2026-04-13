def rescale_intrinsics(global_intrinsics, intrinsics_scales, inverse=False):
    """
    Rescale the intrinsics to the original image size.
    """
    for i in range(len(global_intrinsics)):
        if global_intrinsics[i] is None:
            continue
        x_scale, y_scale, x_offset, y_offset = intrinsics_scales[i]
        if not inverse:
            global_intrinsics[i][0, 0, 2] -= x_offset
            global_intrinsics[i][0, 1, 2] -= y_offset
            global_intrinsics[i][0, 0] /= x_scale
            global_intrinsics[i][0, 1] /= y_scale
        else:
            global_intrinsics[i][0, 0] *= x_scale
            global_intrinsics[i][0, 1] *= y_scale
            global_intrinsics[i][0, 0, 2] += x_offset
            global_intrinsics[i][0, 1, 2] += y_offset

    return global_intrinsics


def keep_inframes(tracks_dict, image_shape_ori, indexes=None):
    if indexes is None:
        indexes = range(len(tracks_dict["indexes"]))

    for idx in indexes:
        index = tracks_dict["indexes"][idx]
        for i, idx_inner in enumerate(index):
            is_valid_x = (tracks_dict["tracks"][idx][0, i, :, 0] >= 0) * (
                tracks_dict["tracks"][idx][0, i, :, 0]
                < image_shape_ori[idx_inner][1]
            )
            is_valid_y = (tracks_dict["tracks"][idx][0, i, :, 1] >= 0) * (
                tracks_dict["tracks"][idx][0, i, :, 1]
                < image_shape_ori[idx_inner][0]
            )

            tracks_dict["vis"][idx][0, i] = (
                tracks_dict["vis"][idx][0, i] * is_valid_x * is_valid_y
            )


# Change from original images to rescaled images
def standardize_query_points(query_points, image_change):
    """
    query_points: (1, K, 2) or (K, 2)
    image_change: (x_scale, y_scale, x_shift, y_shift)
    """
    if query_points.dim() == 3:
        assert query_points.shape[0] == 1

    query_points[..., 0] = (
        query_points[..., 0] * image_change[0] + image_change[2]
    )
    query_points[..., 1] = (
        query_points[..., 1] * image_change[1] + image_change[3]
    )

    return query_points


# Change from rescaled images to the original images
def rescale_tracks_single(tracks, image_change):
    """
    tracks: (1, K, 2) or (K, 2)
    """
    if tracks.dim() == 3:
        assert tracks.shape[0] == 1

    tracks[..., 0] = (tracks[..., 0] - image_change[2]) / image_change[0]
    tracks[..., 1] = (tracks[..., 1] - image_change[3]) / image_change[1]

    return tracks


def rescale_tracks(
    tracks_dict, image_changes_full, indexes=None, reverse=False
):
    """
    Rescale the tracks to the original image size.
    """
    if indexes is None:
        range(len(tracks_dict["indexes"]))

    # image_changes_full: (x_scale, y_scale, x_shift, y_shift)
    for idx in indexes:
        index = tracks_dict["indexes"][idx]
        for i, idx_inner in enumerate(index):
            # rescale the tracks
            if not reverse:
                tracks_dict["tracks"][idx][0, i] = rescale_tracks_single(
                    tracks_dict["tracks"][idx][0, i],
                    image_changes_full[idx_inner],
                )
            else:
                tracks_dict["tracks"][idx][0, i] = standardize_query_points(
                    tracks_dict["tracks"][idx][0, i],
                    image_changes_full[idx_inner],
                )

    if "tracks_virtual" in tracks_dict:
        for idx in indexes:
            index = tracks_dict["indexes"][idx]
            for i, idx_inner in enumerate(index):
                # rescale the tracks
                if not reverse:
                    tracks_dict["tracks_virtual"][idx][0, i] = (
                        rescale_tracks_single(
                            tracks_dict["tracks_virtual"][idx][0, i],
                            image_changes_full[idx_inner],
                        )
                    )
                else:
                    tracks_dict["tracks_virtual"][idx][0, i] = (
                        standardize_query_points(
                            tracks_dict["tracks_virtual"][idx][0, i],
                            image_changes_full[idx_inner],
                        )
                    )

    return tracks_dict
