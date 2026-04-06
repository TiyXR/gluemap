import logging
import os
import torch
import torch.distributed as dist
import pickle
import shutil
from typing import Any

logger = logging.getLogger(__name__)


def init_distributed_mode(args):
    nodist = args.nodist if hasattr(args, "nodist") else False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not nodist:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        logger.info("Not using distributed mode")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"

    # Fall back to TCP sockets when InfiniBand / OFI (libfabric) are unavailable
    # (e.g. CSCS/Alps Slingshot nodes without the AWS OFI NCCL plugin).
    # Safe for inference pipelines where high-throughput collectives are not needed.
    os.environ["NCCL_NET"] = "Socket"
    os.environ["NCCL_IB_DISABLE"] = "1"

    logger.info(f"| distributed init (rank {args.rank}): {args.dist_url}, gpu {args.gpu}")
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """Suppress info-level logging on non-master processes."""
    if not is_master:
        logging.getLogger().setLevel(logging.WARNING)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def all_gather_object_cpu(  # type: ignore
    data: Any,
    tmpdir: None | str = None,
    rank_zero_return_only: bool = True,
    use_system_tmp: bool = False,
) -> list[Any] | None:  # pragma: no cover
    """Share arbitrary picklable data via file system caching.

    Args:
        data: any picklable object.
        tmpdir: Save path for temporary files. If None, safely create tmpdir.
        rank_zero_return_only: if results should only be returned on rank 0.
        use_system_tmp: if use system tmpdir or not.

    Returns:
        list[Any]: list of data gathered from each process.
    """
    rank, world_size = get_rank(), get_world_size()
    if world_size == 1:
        return [data]

    # make tmp dir
    # tmpdir = create_tmpdir(rank, tmpdir, use_system_tmp)
    if os.path.exists(tmpdir):
        logger.warning("tmpdir already exists, removing it.")
    else:
        os.makedirs(tmpdir, exist_ok=True)

    # encode & save
    with open(os.path.join(tmpdir, f"part_{rank}.pkl"), "wb") as f:
        pickle.dump(data, f)
    synchronize()

    if rank_zero_return_only and not rank == 0:
        return None

    # load & decode
    data_list = []
    for i in range(world_size):
        with open(os.path.join(tmpdir, f"part_{i}.pkl"), "rb") as f:
            data_list.append(pickle.load(f))

    # remove dir
    if not rank_zero_return_only:
        # wait for all processes to finish loading before removing tmpdir
        synchronize()
    if rank == 0:
        shutil.rmtree(tmpdir)

    return data_list


def synchronize() -> None:  # pragma: no cover
    """Sync (barrier) among all processes when using distributed training."""
    if not dist.is_available() and dist.is_initialized():
        return
    if get_world_size() == 1:
        return

    # TODO: here, multi GPU is not supported
    # dist.barrier(group=dist.group.WORLD, device_ids=[get_rank()])
    dist.barrier()
