from lightglue import LightGlue, SuperPoint, SIFT, ALIKED
import torch

from gluemap.math.geometry import bilinear_interpolate_value

def get_query_points(
    query_image,
    query_method,
    max_query_num=4096,
    det_thres=0.005,
    sim_score=None,
    bbox=None,
    mask=None, # should be of size (B, H, W)
    expand_ratio=1,
    strict_num=True,
):
    # Run superpoint and sift on the target frame
    # Feel free to modify for your own
    max_query_num_raw = int(max_query_num * expand_ratio)

    methods = query_method.split("+")
    pred_points = []

    for method in methods:
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
            extractor = SIFT(max_num_keypoints=max_query_num_raw).to(query_image.device).eval()
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

        # In VGGT code, it use the rounded keypoints
        query_points = extractor.extract(query_image)["keypoints"]
        pred_points.append(query_points)

    query_points = torch.cat(pred_points, dim=1)
    
    # Remove points outside the mask
    if mask is not None:
        is_valid = bilinear_interpolate_value(mask.unsqueeze(1).float(), query_points).squeeze(-1)
        query_points = query_points[is_valid > 0.5].unsqueeze(0)
    
    if query_points.shape[1] < max_query_num:
        if query_points.shape[1] == 0:
            query_points = torch.rand(1, max_query_num, 2, device=query_image.device)
            return query_points
        # elif len(methods) < 2:
        #     method = "sift+aliked"
        #     return get_query_points(query_image, method, max_query_num_raw, det_thres, sim_score, bbox)

    # plt.clf()
    # plt.imshow(query_image[0].cpu().numpy().transpose(1, 2, 0))
    # plt.scatter(
    #     query_points[0, :, 0].cpu().numpy(), query_points[0, :, 1].cpu().numpy(), s=1
    # )
    # plt.savefig("query_points_raw.png")
    # cmap = plt.get_cmap("viridis")

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
        # Duplicate the points to match the max_query_num
        # Add some random points to the query points
        # query_points = query_points.repeat(1, int(math.ceil(max_query_num / query_points.shape[1])), 1)[:, :max_query_num, :]
        # print(query_points.shape[1])
        query_points = torch.cat(
            [
                query_points,
                torch.cat(
                [torch.rand(
                    1,
                    max_query_num - query_points.shape[1],
                    1,
                    device=query_image.device,
                ) * query_image.shape[-1], torch.rand(
                    1,
                    max_query_num - query_points.shape[1],
                    1,
                    device=query_image.device,
                ) * query_image.shape[-2]],
                dim=-1,
                ),
            ],
            dim=1,
        )
        

    # plt.clf()
    # plt.imshow(query_image[0].cpu().numpy().transpose(1, 2, 0))
    # plt.scatter(
    #     query_points[0, :, 0].cpu().numpy(), query_points[0, :, 1].cpu().numpy(), s=1
    # )
    # plt.savefig("query_points_sampled.png")
    # breakpoint()

    return query_points # since by default, the query points with have a + 0.5 offset

