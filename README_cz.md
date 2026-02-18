cd rllm
rm -rf ~/.cache/uv

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5
export NCCL_GID_INDEX=3
uv sync
## Intallation

```
uv sync

uv pip install "r2e-gym@https://github.com/R2E-Gym/R2E-Gym.git"
uv pip install -e /shared_workspace_mfs/zhilong/mindforge_harness/
uv pip install -e .[verl] 
```

## Prepare data

```
python scripts/data/swe_dataset.py --local_dir <your path>
```

## Run the script


#### On host:
```
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export WANDB_API_KEY=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
export GLOO_PORT_RANGE=20001-21000
GLOO_SOCKET_IFNAME=bond0
ray start --head --min-worker-port 20001 --max-worker-port 21000
```

#### On other machines
```
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
WANDB_API_KEY=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
export GLOO_PORT_RANGE=20001-21000
GLOO_SOCKET_IFNAME=bond0
ray start --address "10.10.110.129:6379" --min-worker-port 20001 --max-worker-port 21000
```

#### Training script
```
examples/swe/train_deepswe_8b.sh
```
Note taht the current script is using one node only. Change the `nnode` to use multiple nodes.

## Docker

```bash
docker create \
  --runtime=nvidia \
  --gpus all \
  --ipc=host \
  --net=host \
  --shm-size="64g" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device=/dev/infiniband \
  --cap-add=SYS_ADMIN \
  -v /shared_workspace_mfs/chengzong/mindforge_harness:/workspace/mindforge_harness \
  -v /shared_workspace_mfs/rllm:/workspace/rllm \
  -v /shared_workspace_mfs:/shared_workspace_mfs \
  --name rllm \
  rllm \
  sleep infinity
docker start rllm
docker exec -it rllm bash
```

And follow the instructions above