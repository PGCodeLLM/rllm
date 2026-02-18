#!/usr/bin/env python3
"""
NCCL / TCPStore distribution test script to reproduce:

  [rank1]: sendBytes failed ... Broken pipe
  ProcessGroupNCCL: Failed to check the "should dump" flag on TCPStore,
  (maybe TCPStore server has shut down too early), with error: Broken pipe

Run with torchrun (single node, 2 GPUs):
  torchrun --nproc_per_node=2 scripts/test_nccl_distributed.py

Reproduce the error (rank 0 exits early, so TCPStore server dies; other ranks get Broken pipe):
  torchrun --nproc_per_node=2 scripts/test_nccl_distributed.py --early-exit-rank 0

Use NCCL backend (default if CUDA available) or force Gloo (CPU):
  torchrun --nproc_per_node=2 scripts/test_nccl_distributed.py --backend gloo
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description="NCCL/TCPStore distributed test")
    parser.add_argument(
        "--early-exit-rank",
        type=int,
        default=None,
        metavar="R",
        help="If set, this rank exits immediately after init_process_group (no barrier). "
        "Reproduces TCPStore 'Broken pipe' on other ranks when R=0 (store server exits).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["nccl", "gloo"],
        help="Process group backend. Default: nccl if CUDA else gloo.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Use CPU (gloo) even if CUDA is available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("Must be run with torchrun, e.g.: torchrun --nproc_per_node=2 scripts/test_nccl_distributed.py")
        sys.exit(1)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    if args.backend is None:
        backend = "nccl" if use_cuda else "gloo"
    else:
        backend = args.backend

    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

    print(f"[rank{rank}] Initializing process group backend={backend} world_size={world_size} device={device}", flush=True)

    dist.init_process_group(backend=backend, init_method="env://")

    # Reproduce "TCPStore server has shut down too early" / Broken pipe:
    # The process that runs the TCPStore server (rank 0 with env://) exits without
    # barrier/destroy. Other ranks then touch the store (e.g. barrier) and get Broken pipe.
    if args.early_exit_rank is not None and rank == args.early_exit_rank:
        print(f"[rank{rank}] EARLY EXIT (no barrier) to trigger Broken pipe on other ranks", flush=True)
        os._exit(0)  # no barrier, no destroy; TCPStore server (this process) goes away

    # Other ranks: do a barrier. If early-exit rank was 0, store is dead -> Broken pipe.
    try:
        dist.barrier()
        print(f"[rank{rank}] barrier() OK", flush=True)
    except Exception as e:
        print(f"[rank{rank}] barrier() failed: {e}", flush=True)
        dist.destroy_process_group()
        raise

    # One collective so NCCL is used (triggers "should dump" check on store in some versions)
    x = torch.ones(2, device=device)
    try:
        dist.all_reduce(x)
        print(f"[rank{rank}] all_reduce OK", flush=True)
    except Exception as e:
        print(f"[rank{rank}] all_reduce failed: {e}", flush=True)
        dist.destroy_process_group()
        raise

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank{rank}] Done.", flush=True)


if __name__ == "__main__":
    main()
