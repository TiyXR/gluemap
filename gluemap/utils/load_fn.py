import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TF


def load_and_preprocess_images_inner(
    images_ori, image_size=518, force_square=True, patch_size=14
):
    # Check for empty list
    if len(images_ori) == 0:
        raise ValueError("At least 1 image is required")

    assert image_size % patch_size == 0, (
        "Image size must be divisible by patch_size for compatibility with model requirements"
    )

    images = []
    shapes = set()

    if force_square:
        shapes.add((image_size, image_size))  # Add square shape for padding

    images_change = []  # (scale_x, scale_y, x_offset, y_offset)
    # First process all images and collect their shapes
    for i in range(len(images_ori)):
        img = images_ori[i].clone()

        # width, height = img.size
        height, width = img.shape[-2:]

        if width > height:
            new_width = image_size

            # Calculate height maintaining aspect ratio, divisible by patch_size
            new_height = (
                round(height * (new_width / width) / patch_size) * patch_size
            )

        else:
            new_height = image_size

            # Calculate width maintaining aspect ratio, divisible by patch_size
            new_width = (
                round(width * (new_height / height) / patch_size) * patch_size
            )
            shapes.add(
                (image_size, image_size)
            )  # since VGGT does not support portrait images, always pad them

        # Resize with new dimensions (width, height)
        img = F.interpolate(
            img.unsqueeze(0),
            size=(int(new_height), int(new_width)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        images_change.append([new_width / width, new_height / height, 0, 0])

    # Check if we have different shapes
    if len(shapes) > 1:
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for i, img in enumerate(images):
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
                images_change[i][-2] += pad_left
                images_change[i][-1] += pad_top

            padded_images.append(img)

        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(images_ori) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images, images_ori, images_change


def load_and_preprocess_images(
    image_path_list, image_size=518, force_square=True, patch_size=14
):
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    assert image_size % patch_size == 0, (
        "Image size must be divisible by patch_size for compatibility with model requirements"
    )

    to_tensor = TF.ToTensor()

    images_ori = []
    images_change = []  # (scale_x, scale_y, x_offset, y_offset)
    # First process all images and collect their shapes
    for image_path in image_path_list:
        img = Image.open(image_path).convert("RGB")
        images_ori.append(to_tensor(img.copy()))

    return load_and_preprocess_images_inner(
        images_ori,
        image_size=image_size,
        force_square=force_square,
        patch_size=patch_size,
    )


def load_and_preprocess_images_1024(images_ori):

    N = len(images_ori)
    # Check for empty list
    if len(images_ori) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    # images_ori = []
    images_change = []  # (scale_x, scale_y, x_offset, y_offset)
    # First process all images and collect their shapes
    for i in range(N):
        # img = Image.open(image_path).convert("RGB")
        # images_ori.append(to_tensor(img.copy()))
        img = images_ori[i].clone()

        # width, height = img.size
        height, width = img.shape[1:3]

        if width > height:
            new_width = 1024

            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width))

        else:
            new_height = 1024

            # Calculate width maintaining aspect ratio, divisible by 14
            new_width = round(width * (new_height / height))

        shapes.add(
            (1024, 1024)
        )  # since VGGT does not support portrait images, always pad them

        # Resize with new dimensions (width, height)
        img = F.interpolate(
            img.unsqueeze(0),
            size=(int(new_height), int(new_width)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # # Center crop height if it's larger than 518
        # start_y = 0
        # if new_height > 518:
        #     start_y = (new_height - 518) // 2
        #     img = img[:, start_y : start_y + 518, :]

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        images_change.append([new_width / width, new_height / height, 0, 0])

    # Check if we have different shapes
    # In theory our model can also work well with different shapes

    if len(shapes) > 1:
        # print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for i, img in enumerate(images):
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
                images_change[i][-2] += pad_left
                images_change[i][-1] += pad_top

            padded_images.append(img)

        images = padded_images

    images = torch.stack(images)  # concatenate images

    return images, images_change


def calculate_image_shapes(images_shape_ori, new_shape_hw):
    images_change = []

    max_new_shape = max(new_shape_hw[0], new_shape_hw[1])

    if new_shape_hw[1] == 512:
        patch_size = 16
    else:
        patch_size = 14

    for image_shape in images_shape_ori:
        height, width = image_shape
        if width > height:
            new_width = max_new_shape

            # Calculate height maintaining aspect ratio, divisible by patch_size
            new_height = (
                round(height * (new_width / width) / patch_size) * patch_size
            )

        else:
            new_height = max_new_shape

            # Calculate width maintaining aspect ratio, divisible by patch_size
            new_width = (
                round(width * (new_height / height) / patch_size) * patch_size
            )

        images_change.append([new_width / width, new_height / height, 0, 0])

        h_padding = new_shape_hw[0] - new_height
        w_padding = new_shape_hw[1] - new_width

        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left

            images_change[-1][-2] += pad_left
            images_change[-1][-1] += pad_top

    return images_change


def unify_image_sizes(images_ori, images_change):
    N = len(images_ori)
    # Check for empty list
    if len(images_ori) == 0:
        raise ValueError("At least 1 image is required")

    max_width = 0
    max_height = 0
    for i in range(N):
        height, width = images_ori[i].shape[1:3]

        max_width = max(max_width, width)
        max_height = max(max_height, height)

    for i in range(N):
        img = F.interpolate(
            images_ori[i].unsqueeze(0),
            size=(int(max_height), int(max_width)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        images_change[i][0] = (
            images_ori[i].shape[2] / max_width * images_change[i][0]
        )
        images_change[i][1] = (
            images_ori[i].shape[1] / max_height * images_change[i][1]
        )
        images_ori[i] = img

    return images_ori, images_change
