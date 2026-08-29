import argparse
import logging
import os
import pickle
import shutil
from typing import Any

import torch
import torch.distributed as dist

from gluemap.utils.model_loader import load_models

logger = logging.getLogger(__name__)


def init_distributed_mode(args: argparse.Namespace) -> None:
    """Initialize ``torch.distributed`` from environment variables.

    Reads ``RANK``, ``WORLD_SIZE``, and ``LOCAL_RANK`` from the environment
    (the standard handshake set up by ``torchrun``/Slurm) and calls
    ``init_process_group`` with the NCCL backend. ``args`` is mutated to
    record the chosen ``rank``, ``world_size``, ``gpu``, ``dist_backend``,
    and ``distributed`` flag.

    The function bails out early — leaving ``args.distributed = False`` —
    when ``args.cpu`` is set, when ``args.nodist`` is truthy, or when the
    expected env vars are absent.

    Sets ``NCCL_NET=Socket`` and ``NCCL_IB_DISABLE=1`` so NCCL falls back to
    TCP sockets on clusters without the OFI/InfiniBand stack (e.g. CSCS
    Alps without the AWS OFI NCCL plugin).
    """
    if getattr(args, "cpu", False):
        logger.info(
            "Running on CPU (--cpu flag set), distributed mode disabled"
        )
        setup_for_distributed(is_master=True)
        args.distributed = False
        return

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

    # Fall back to TCP sockets when InfiniBand / OFI (libfabric) are
    # unavailable (e.g. CSCS/Alps Slingshot nodes without the AWS OFI NCCL
    # plugin). Safe for inference pipelines where high-throughput collectives
    # are not needed.
    os.environ["NCCL_NET"] = "Socket"
    os.environ["NCCL_IB_DISABLE"] = "1"

    logger.info(
        f"| distributed init (rank {args.rank}): {args.dist_url}, "
        f"gpu {args.gpu}"
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master: bool) -> None:
    """Suppress info-level logging on non-master processes."""
    if not is_master:
        logging.getLogger().setLevel(logging.WARNING)


def is_dist_avail_and_initialized() -> bool:
    """Return ``True`` only when ``torch.distributed`` is available and
    initialized."""
    if not dist.is_available():
        return False
    return dist.is_initialized()


def get_world_size() -> int:
    """Return the world size (1 outside distributed mode)."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Return the current rank (0 outside distributed mode)."""
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

    if rank_zero_return_only and rank != 0:
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


def init_distributed(
    args: argparse.Namespace,
) -> tuple[int, int, torch.device, torch.dtype]:
    """Initialize distributed mode and return rank, world_size, device, and
    dtype."""
    init_distributed_mode(args)
    rank = get_rank()
    world_size = get_world_size()

    # Cached post-processing does not need a feed-forward checkpoint merely
    # to discover the device. Individual stages still load their model if
    # the corresponding cache is absent.
    if getattr(args, "force_load", False):
        device = torch.device(
            "cpu"
            if getattr(args, "cpu", False) or not torch.cuda.is_available()
            else "cuda"
        )
    else:
        _, device = load_models(args, keys=set())
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16

    return rank, world_size, device, dtype
