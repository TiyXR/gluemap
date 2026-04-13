import torch


def get_query_points_from_extractors(
    query_image,
    extractors,
    max_query_num=4096,
    strict_num=True,
):
    """Extract keypoints from a query image using pre-built feature extractors.

    Runs each extractor on ``query_image``, concatenates the detected keypoints,
    and returns exactly ``max_query_num`` points when ``strict_num`` is True by
    either randomly subsampling or padding with random points.

    Args:
        query_image: Input image tensor of shape (1, C, H, W).
        extractors: Iterable of lightglue-style extractors. Each must expose
            ``extract(image)["keypoints"]`` returning a tensor of shape (1, N, 2).
        max_query_num: Target number of keypoints to return.
        strict_num: If True, pad with random points when fewer than ``max_query_num``
            keypoints are detected.

    Returns:
        Tensor of shape (1, N, 2) containing (x, y) keypoint coordinates, where
        N equals ``max_query_num`` when ``strict_num`` is True.
    """
    pred_points = [
        extractor.extract(query_image)["keypoints"] for extractor in extractors
    ]
    query_points = torch.cat(pred_points, dim=1)

    if query_points.shape[1] < max_query_num:
        if query_points.shape[1] == 0:
            query_points = torch.rand(
                1, max_query_num, 2, device=query_image.device
            )
            return query_points

    if query_points.shape[1] > max_query_num:
        random_point_indices = torch.randperm(query_points.shape[1])[
            :max_query_num
        ]
        query_points = query_points[:, random_point_indices, :]
    # If we required the strict number of points, we need to add some random points
    elif strict_num:
        # Add some random points to the query points
        num_pad = max_query_num - query_points.shape[1]
        rand_points = torch.rand(1, num_pad, 2, device=query_image.device)
        rand_points[..., 0] *= query_image.shape[-1]  # width
        rand_points[..., 1] *= query_image.shape[-2]  # height
        query_points = torch.cat([query_points, rand_points], dim=1)

    return query_points  # since by default, the query points with have a + 0.5 offset
