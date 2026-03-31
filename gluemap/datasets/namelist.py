from e2esfm_refactor.datasets.base_dataset import *
import imagesize

import h5py

from experiments.experiment_utils import get_datasets
from e2esfm.pipeline.establish_cluster import establish_neighbors_rig_plain

from tqdm import tqdm
import pickle
import h5py
from e2esfm.pipeline.load_fn import (
    load_and_preprocess_images_1024,
    load_image_from_h5py,
)
from e2esfm_refactor.datasets.utils import convert_pt3d_RT_to_opencv

from torchvision.transforms import ToPILImage
import shutil


class NameListPairs(DemoBaseTwoViewDataset):
    def __init__(self, args, patch_size=16):
        super().__init__(args)
        self.patch_size = patch_size
        if patch_size == 14:
            self.image_size = 518
        else:
            self.image_size = 512

        if (
            args.dataset_category.startswith("lamar")
            and len(args.image_list_short) == 0
        ):
            self._get_lamar_result(args)
        elif args.dataset_category == "co3d":
            self._get_co3d_result(args)
        else:
            # Load all the images, and generate SALAD descriptors for matching
            self.images_list = args.image_list_short

            # img_list = sorted(glob.glob(args.image_ori_path + "/*/*") + glob.glob(args.image_ori_path + "/*"))
            img_list = sorted(glob.glob(args.image_ori_path + "/**", recursive=True))
            # keep only the files
            img_list = [x for x in img_list if os.path.isfile(x)]

            # if the short list is not provided, it should mean that all the images are used
            if (
                not self.images_list
                or len(self.images_list) == 0
                or (len(self.images_list) == 1 and self.images_list[0] == "")
            ):
                self.images_list = [
                    x.replace(args.image_ori_path, "").strip("/") for x in img_list
                ]

            self.image_ori_path = [
                args.image_ori_path for _ in range(len(self.images_list))
            ]

            # if salad descriptors are available, use them to construct pairs, else, use exhaustive matching
            self._construct_pairs(args, img_list)

            # Load ground truth rotations and centers
            self._load_groundtruth(args)

        # get the image size of all the images and preload images if necessary
        self._preload_images(args)

        # prepare camera model
        self._set_camera_model(args)

    def _construct_pairs(self, args, img_list):
        # assume that SALAD descriptors are extracted already
        descriptors_path = os.path.join(args.curr_processed, "salad_descriptors.pt")
        args.num_neighbors = 100

        print("Number of neighbors for retrieval:", args.num_neighbors)

        if os.path.exists(descriptors_path):
            descriptors_db = torch.load(descriptors_path)
            # find the corresponding images and only keep the short list
            image_indexes = [
                img_list.index(os.path.join(args.image_ori_path, img))
                for img in self.images_list
            ]
            descriptors_db = descriptors_db[image_indexes]

            embed_size = descriptors_db.shape[1]
            faiss_index = faiss.IndexFlatL2(embed_size)
            faiss_index.add(descriptors_db)  # add vectors to the index

            num_neighbors = min(args.num_neighbors, len(self.images_list) - 1)
            distance_extend, predictions_extend = faiss_index.search(
                descriptors_db, num_neighbors + 1
            )
        else:
            # consider the case for exhaustive matching
            predictions_extend = np.tile(
                np.arange(0, len(self.images_list)).reshape(1, -1),
                (len(self.images_list), 1),
            )

        # # sample 3-4 times
        # if args.dataset_category == "imc" or args.dataset_category == "co3d":
        #     args.seed_frequency = int(np.ceil(len(self.images_list) / 3))
        # else:
        #     args.seed_frequency = 1
        args.seed_frequency = 1
        # Construct the pairs
        chosen_indexes = np.arange(0, len(self.images_list), args.seed_frequency)
        pairs = np.stack(
            [
                np.tile(chosen_indexes[..., None], (1, predictions_extend.shape[1])),
                predictions_extend[chosen_indexes],
            ],
            axis=-1,
        ).reshape(-1, 2)

        # collect back and force
        pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
        pairs = pairs[
            pairs[:, 0] < pairs[:, 1]
        ]  # only keep pairs where the first index is less than the second
        pairs = np.unique(pairs, axis=0)  # remove duplicates

        self.pairs = pairs

        # prepare pairs for sift
        self.sift_pairs = None
        if (
            args.dataset_category == "imc"
            or args.dataset_category == "co3d"
            or args.dataset_category.startswith("ETH3D")
        ):
            # exhaustive pairs for imc and co3d
            self.sift_pairs = np.stack(
                [
                    np.arange(0, len(self.images_list)),
                    np.arange(0, len(self.images_list)),
                ],
                axis=-1,
            )
            chosen_indexes = np.arange(len(self.images_list))
            pairs = np.stack(
                [
                    np.tile(
                        chosen_indexes[..., None], (1, predictions_extend.shape[1])
                    ),
                    predictions_extend[chosen_indexes],
                ],
                axis=-1,
            ).reshape(-1, 2)
            self.sift_pairs = pairs
        else:
            # otherwise, use the same as the original pairs
            self.sift_pairs = self.pairs

    def _load_groundtruth(self, args):
        dataset = args.dataset
        # different method for loading intrinsics
        # category 1: ground truth in COLMAP format
        if args.dataset_category in self.category_with_colmap_gt:
            if args.dataset_category.startswith("lamar"):
                rotations_gt, centers_gt, intrinsics_mapping_gt, intrinsics_gt = (
                    load_groundtruth(
                        args,
                        dataset.split("__")[0],
                        self.images_list,
                        return_intrinsics=True,
                    )
                )
            else:
                rotations_gt, centers_gt, intrinsics_mapping_gt, intrinsics_gt = (
                    load_groundtruth(
                        args, dataset, self.images_list, return_intrinsics=True
                    )
                )
            self.intrinsics_mapping = intrinsics_mapping_gt
            # self.intrinsics_gt = {intrinsics_mapping_gt[i]: torch.from_numpy(intrinsics_gt[intrinsics_mapping_gt[i]]).unsqueeze(0) for i in range(len(self.images_list))}
            # self.intrinsics_gt = [self.intrinsics_gt[i] for i in sorted(self.intrinsics_gt.keys())]
            self.intrinsics_gt = [
                torch.from_numpy(x).unsqueeze(0) for x in intrinsics_gt
            ]
            # Image are resized by half for ETH3D datasets
            if args.dataset_category.startswith("ETH3D"):
                for i in range(len(self.intrinsics_gt)):
                    self.intrinsics_gt[i][..., 0, :] /= 2
                    self.intrinsics_gt[i][..., 1, :] /= 2

        # category 2: co3d
        elif args.dataset_category == "co3d":
            rotations_gt = []
            centers_gt = []
            # replace the ".jpg" with ".npz" in the image paths
            for i in range(len(self.images_list)):
                meta_path = self.images_list[i].replace(".jpg", ".npz")
                meta_path = os.path.join(args.image_ori_path, meta_path)

                input_metadata = np.load(meta_path)
                camera_pose = input_metadata["camera_pose"].astype(np.float64)
                intrinsics = input_metadata["camera_intrinsics"].astype(np.float64)
                rotations_gt.append(camera_pose[:3, :3].T)
                centers_gt.append(camera_pose[:3, 3])

            # assume all images are using the same intrinsics
            self.intrinsics_mapping = {i: 0 for i in range(len(self.images_list))}

        # category 3: imc
        elif args.dataset_category == "imc":
            rotations_gt = []
            centers_gt = []
            intrinsics_gt = []
            for i in range(len(self.images_list)):
                meta_path = "calibration_" + self.images_list[i].replace(".jpg", ".h5")
                meta_path = os.path.join(
                    args.image_ori_path.replace("/images", "/calibration"), meta_path
                )

                file = h5py.File(meta_path, "r")
                intrinsics = file["K"][()]

                R = file["R"][()]
                T = file["T"][()].reshape(3, 1)

                rotations_gt.append(R)
                centers_gt.append((-R.T @ T).reshape(3))
                intrinsics_gt.append(intrinsics)
                file.close()

            # assume all images are using different intrinsics
            self.intrinsics_mapping = {i: i for i in range(len(self.images_list))}
            self.intrinsics_gt = [
                torch.from_numpy(x).unsqueeze(0) for x in intrinsics_gt
            ]

        self.rotations_gt = {i: rotations_gt[i] for i in range(len(rotations_gt))}
        self.centers_gt = {i: centers_gt[i] for i in range(len(centers_gt))}

    def _preload_images(self, args):
        # get the image size of all the images
        if args.dataset_category != "co3d":
            images_shape_ori = []
            for img_path in tqdm(self.images_list):
                img_size = imagesize.get(os.path.join(args.image_ori_path, img_path))
                images_shape_ori.append(
                    img_size[::-1]
                )  # (width, height) -> (height, width)

            self.images_shape_ori = images_shape_ori

        # if the number of images is fewer than 200 images, directly load the images
        if len(self.images_list) < 200:
            self.has_preloaded = True
            if args.dataset_category != "co3d":
                self.images, self.images_ori, self.images_change = (
                    load_and_preprocess_images_ori(
                        [
                            os.path.join(self.image_ori_path[i], self.images_list[i])
                            for i in range(len(self.images_list))
                        ],
                        image_size=self.image_size,
                        patch_size=self.patch_size,
                    )
                )
            else:
                self.images, self.images_ori, self.images_change = load_image_from_h5py(
                    args.image_ori_path,
                    self.images_list,
                    image_size=self.image_size,
                    patch_size=self.patch_size,
                )

                self.images_shape_ori = [
                    self.images_ori[i].shape[-2:] for i in range(len(self.images_ori))
                ]

                # Remove the original images
                if os.path.exists(args.curr_path + "/images"):
                    shutil.rmtree(args.curr_path + "/images")

                os.makedirs(args.curr_path + "/images", exist_ok=True)
                for i in range(len(self.images_list)):
                    image_pil = ToPILImage()(self.images_ori[i])
                    image_pil.save(
                        os.path.join(args.curr_path, "images", self.images_list[i])
                    )

                args.image_ori_path = os.path.join(args.curr_path, "images")

            self.images_1024, self.images_change_1024 = load_and_preprocess_images_1024(
                self.images_ori
            )
        else:
            print("images are not preloaded!")

        print("preload image done")

    def _set_camera_model(self, args):
        # if args.dataset_category in ["ETH3D", "co3d"]:
        #     self.camera_model = "PINHOLE"
        # elif args.dataset_category in ["imc"]:
        #     # self.camera_model = "SIMPLE_RADIAL"
        #     self.camera_model = "SIMPLE_PINHOLE"
        # elif args.dataset_category.startswith("lamar"):
        #     self.camera_model = "PINHOLE"
        # else:
        #     self.camera_model = "SIMPLE_PINHOLE"
        self.camera_model = "SIMPLE_PINHOLE"

        print("set camera model done")

    def _get_lamar_result(self, args):

        # Read in the images list
        datasets = get_datasets(args)

        images_list_all = []
        image_ori_path_all = []
        desctriptors_all = []
        pairs_ori = []
        neighbors_pre = []
        image_count = 0
        intrinsics_count = 0

        self.rotations_gt = {}
        self.centers_gt = {}
        self.intrinsics_mapping = {}

        args.num_neighbors = 30
        args.num_neighbors_retrieval = 30

        starting_indexes = []
        self.intrinsics_gt = []
        for dataset in datasets:
            if dataset.startswith("ios"):
                args.num_neighbors = 30
            else:
                args.num_neighbors = 64

            starting_indexes.append(image_count)
            images_list_short, image_ori_path = load_images(
                args, dataset, None, return_ori_path=True, load_images=False
            )

            pos = len(args.image_ori_path)
            images_list_all.append(
                [
                    image_ori_path[pos + 1 :].strip("/") + "/" + x
                    for x in images_list_short
                ]
            )
            # image_ori_path_all.append(image_ori_path)
            image_ori_path_all.append(args.image_ori_path)

            prefix = (
                args.dataset_category
                if args.dataset_category != "ETH3D"
                else args.dataset_category + "_" + args.mode
            )
            dir_write = os.path.join(args.processed_path, prefix, dataset)
            desctriptors_all.append(
                torch.load(os.path.join(dir_write, "salad_descriptors.pt"))
            )
            neighbors = establish_neighbors_rig_plain(
                images_list_short, num_neighbors=args.num_neighbors
            )
            rotations_gt, centers_gt, intrinsics_mapping, intrinsics_gt = (
                load_groundtruth(
                    args, dataset, images_list_short, return_intrinsics=True
                )
            )

            rotations_gt = {
                i + image_count: rotations_gt[i] for i in range(len(rotations_gt))
            }
            centers_gt = {
                i + image_count: centers_gt[i] for i in range(len(centers_gt))
            }

            intrinsics_mapping = {
                i + image_count: intrinsics_mapping[i] + intrinsics_count
                for i in intrinsics_mapping.keys()
            }

            self.rotations_gt = self.rotations_gt | rotations_gt
            self.centers_gt = self.centers_gt | centers_gt
            self.intrinsics_mapping = self.intrinsics_mapping | intrinsics_mapping
            self.intrinsics_gt.extend(intrinsics_gt)

            neighbors = torch.tensor(neighbors, dtype=torch.int64)
            neighbors = neighbors + image_count
            # if the number of neighbor is too few, pad it with the last value
            # it is fine since only unique value will remain
            if neighbors.shape[1] < args.num_neighbors + 1:
                diff = args.num_neighbors + 1 - neighbors.shape[1]
                neighbors = torch.cat(
                    [neighbors, neighbors[:, -1:].repeat(1, diff)], dim=1
                )
            neighbors_pre.append(neighbors)

            image_count += len(images_list_short)
            intrinsics_count += len(set(intrinsics_mapping.values()))

        self.intrinsics_gt = [
            torch.from_numpy(x).unsqueeze(0) for x in self.intrinsics_gt
        ]

        args.num_neighbors = 64
        # neighbors_local = torch.cat(neighbors_pre, dim=0).cpu().numpy()
        neighbors_local = [
            x[i].cpu().numpy() for x in neighbors_pre for i in range(x.shape[0])
        ]

        N_total = len(self.rotations_gt)

        self.images_list = [x for y in images_list_all for x in y]
        self.image_ori_path = [
            x
            for i, x in enumerate(image_ori_path_all)
            for _ in range(len(images_list_all[i]))
        ]

        descritptors_db = torch.cat(desctriptors_all, dim=0)
        embed_size = descritptors_db.shape[1]
        faiss_index = faiss.IndexFlatL2(embed_size)
        faiss_index.add(descritptors_db)  # add vectors to the index

        # Only run inference on every 10th image
        chosen_indexes = np.arange(0, descritptors_db.shape[0], 1)
        # print("Starting retrieval...")
        # start = time.time()
        # sampled_neighbors = [set(neighbors_local[i].tolist()) for i in range(neighbors_local.shape[0])]
        sampled_neighbors = [
            set(neighbors_local[i].tolist()) for i in range(len(neighbors_local))
        ]
        distance_extend, predictions_extend = faiss_index.search(
            descritptors_db[chosen_indexes],
            args.num_neighbors_retrieval + args.num_neighbors + 1,
        )

        print("Retrieval for all images done")

        # exclude itself
        # distance_extend = distance_extend[:, 1:]
        predictions_extend = predictions_extend[:, 1:]

        # exclude the ones that are have already included in neighbors_local
        chosen_indexes_item = [
            [
                j
                for j, x in enumerate(predictions_extend[i].tolist())
                if x not in sampled_neighbors[i]
            ][: args.num_neighbors_retrieval]
            for i in range(len(predictions_extend))
        ]

        # print([len(x) for x in chosen_indexes_item])

        # breakpoint()
        predictions_extend = np.array(
            [
                predictions_extend[i][chosen_indexes_item[i]]
                for i in range(len(predictions_extend))
            ]
        )
        # distance_extend = np.array([distance_extend[i][chosen_indexes[i]] for i in range(len(distance_extend))])

        # pairs_local = np.stack([np.tile(np.arange(N_total)[..., None], (1, neighbors_local.shape[1])), neighbors_local], axis=-1).reshape(-1, 2)
        pairs_local = np.concatenate(
            [
                np.stack(
                    [
                        np.ones(neighbors_local[i].shape, dtype=int) * i,
                        neighbors_local[i],
                    ],
                    axis=-1,
                )
                for i in range(len(neighbors_local))
            ],
            axis=0,
        )

        # Construct the pairs
        pairs_retrieval = np.stack(
            [
                np.tile(chosen_indexes[..., None], (1, predictions_extend.shape[1])),
                predictions_extend,
            ],
            axis=-1,
        ).reshape(-1, 2)

        pairs = np.array(
            sorted(
                list(
                    set(tuple(x) for x in pairs_retrieval).union(
                        tuple(x) for x in pairs_local
                    )
                )
            )
        )

        # collect back and force
        pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
        pairs = pairs[
            pairs[:, 0] < pairs[:, 1]
        ]  # only keep pairs where the first index is less than the second
        pairs = np.unique(pairs, axis=0)  # remove duplicates

        self.pairs = pairs
        self.sift_pairs = self.pairs

    def _get_co3d_result(self, args):

        # Load all the images, and generate SALAD descriptors for matching
        self.images_list = args.image_list_short
        self.image_ori_path = [
            args.image_ori_path for _ in range(len(self.images_list))
        ]

        # if salad descriptors are available, use them to construct pairs, else, use exhaustive matching
        self._construct_pairs(args, self.images_list)

        # Load ground truth
        self.intrinsics_mapping = {i: 0 for i in range(len(self.images_list))}

        ori_path = "/".join(args.image_ori_path.split("/")[:-1])
        category, seq_name = args.image_ori_path.split("/")[-1].split(".")[:2]
        # self.rotations_gt = {
        data = pickle.load(open(f"{ori_path}/co3d_gt_{category}.pkl", "rb"))[seq_name]

        Rs = [np.array(data[i]["R"]) for i in range(len(data))]
        Ts = [np.array(data[i]["T"]) for i in range(len(data))]

        extri = [convert_pt3d_RT_to_opencv(Rs[i], Ts[i]) for i in range(len(Rs))]
        self.rotations_gt = {i: extri[i][:3, :3] for i in range(len(data))}
        self.centers_gt = {
            i: -(extri[i][:3, :3].T @ extri[i][:, 3:]).reshape(
                3,
            )
            for i in range(len(data))
        }
