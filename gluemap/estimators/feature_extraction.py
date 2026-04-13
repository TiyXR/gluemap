from lightglue import SuperPoint, SIFT, ALIKED
import torch

from gluemap.math.geometry import bilinear_interpolate_value


def get_query_points_from_extractors(
    query_image,
    extractors,
    max_query_num=4096,
    sim_score=None,
    bbox=None,
    mask=None,  # should be of size (B, H, W)
    strict_num=True,
):
    """Extract keypoints from a query image using pre-built feature extractors.

    Runs each extractor on ``query_image``, concatenates the detected keypoints,
    optionally filters them by a binary mask, and returns exactly ``max_query_num``
    points when ``strict_num`` is True by either subsampling (weighted by
    ``sim_score`` if provided) or padding with random points.

    Args:
        query_image: Input image tensor of shape (1, C, H, W).
        extractors: Iterable of lightglue-style extractors. Each must expose
            ``extract(image)["keypoints"]`` returning a tensor of shape (1, N, 2).
        max_query_num: Target number of keypoints to return.
        sim_score: Optional similarity score tensor used to weight keypoint sampling
            when more points are detected than ``max_query_num``.
        bbox: Optional bounding box tensor of shape (1, 1, 4) as (x1, y1, x2, y2).
            Points outside the box receive zero sampling weight.
        mask: Optional binary mask of shape (B, H, W). Points where the mask value
            is below 0.5 are discarded.
        strict_num: If True, pad with random points when fewer than ``max_query_num``
            keypoints are detected.

    Returns:
        Tensor of shape (1, N, 2) containing (x, y) keypoint coordinates, where
        N equals ``max_query_num`` when ``strict_num`` is True.
    """
    pred_points = [extractor.extract(query_image)["keypoints"] for extractor in extractors]
    query_points = torch.cat(pred_points, dim=1)

    # Remove points outside the mask
    if mask is not None:
        is_valid = bilinear_interpolate_value(
            mask.unsqueeze(1).float(), query_points
        ).squeeze(-1)
        query_points = query_points[is_valid > 0.5].unsqueeze(0)

    if query_points.shape[1] < max_query_num:
        if query_points.shape[1] == 0:
            query_points = torch.rand(1, max_query_num, 2, device=query_image.device)
            return query_points

    if query_points.shape[1] > max_query_num:
        # Generate the score for the weights
        if sim_score is not None:
            bins = int(sim_score.shape[1] ** 0.5)
            bin_width = (query_image.shape[-1] + 1) / bins
            img_width = query_image.shape[-1]
            # TODO: check whether x should be swapped with y
            scores = sim_score[
                0,
                (
                    torch.clamp(query_points[0, :, 1], 0, img_width) // bin_width * bins
                    + torch.clamp(query_points[0, :, 0], 0, img_width) // bin_width
                ).long(),
            ].sum(dim=1)
            if bbox is not None:
                scores = torch.where(
                    (
                        (query_points[0, :, 0] >= bbox[0, 0, 0])
                        * (query_points[0, :, 1] >= bbox[0, 0, 1])
                        * (query_points[0, :, 0] <= bbox[0, 0, 2])
                        * (query_points[0, :, 1] <= bbox[0, 0, 3])
                    ),
                    scores,
                    torch.zeros_like(scores),
                )
            random_point_indices = torch.multinomial(
                scores, max_query_num, replacement=False
            )
        else:
            random_point_indices = torch.randperm(query_points.shape[1])[:max_query_num]
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


def get_query_points(
    query_image,
    query_method,
    max_query_num=4096,
    det_thres=0.005,
    sim_score=None,
    bbox=None,
    mask=None,  # should be of size (B, H, W)
    expand_ratio=1,
    strict_num=True,
):
    """Extract keypoints from a query image using one or more feature detectors.

    Builds the SuperPoint/SIFT/ALIKED extractors named in ``query_method`` and
    delegates to :func:`get_query_points_from_extractors` for detection, filtering,
    and subsampling. Callers that already hold extractors should call that helper
    directly to avoid re-instantiating them on every invocation.

    Args:
        query_image: Input image tensor of shape (1, C, H, W).
        query_method: Detector name(s) joined by "+", e.g. "sp+sift".
        max_query_num: Target number of keypoints to return.
        det_thres: Detection threshold forwarded to the feature extractor.
        sim_score: Optional similarity score tensor used to weight keypoint sampling
            when more points are detected than ``max_query_num``.
        bbox: Optional bounding box tensor of shape (1, 1, 4) as (x1, y1, x2, y2).
            Points outside the box receive zero sampling weight.
        mask: Optional binary mask of shape (B, H, W). Points where the mask value
            is below 0.5 are discarded.
        expand_ratio: Multiplier on ``max_query_num`` passed to the extractor to
            over-detect before subsampling.
        strict_num: If True, pad with random points when fewer than ``max_query_num``
            keypoints are detected.

    Returns:
        Tensor of shape (1, N, 2) containing (x, y) keypoint coordinates, where
        N equals ``max_query_num`` when ``strict_num`` is True.
    """
    max_query_num_raw = int(max_query_num * expand_ratio)

    extractors = []
    for method in query_method.split("+"):
        if "sp" in method:
            extractor = (
                SuperPoint(
                    max_num_keypoints=max_query_num_raw, detection_threshold=det_thres
                )
                .to(query_image.device)
                .eval()
            )
        elif "sift" in method:
            # extractor = SIFT(max_num_keypoints=max_query_num_raw, backend="pycolmap_cpu").cuda().eval()
            extractor = (
                SIFT(max_num_keypoints=max_query_num_raw).to(query_image.device).eval()
            )
        elif "aliked" in method:
            extractor = (
                ALIKED(
                    max_num_keypoints=max_query_num_raw, detection_threshold=det_thres
                )
                .to(query_image.device)
                .eval()
            )
        else:
            raise NotImplementedError(f"query method {method} is not supprted now")
        extractors.append(extractor)

    return get_query_points_from_extractors(
        query_image,
        extractors,
        max_query_num=max_query_num,
        sim_score=sim_score,
        bbox=bbox,
        mask=mask,
        strict_num=strict_num,
    )
