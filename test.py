# modded-nanogpt/test.py
import torch
import torch.distributed as dist
import os

def main():
    # Initialize distributed environment
    dist.init_process_group(backend='nccl')
    
    # Retrieve environment variables
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    
    # Set device for the current process
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    print(f"Process {ddp_rank} using device: {device}")
    
    master_process = (ddp_rank == 0)  # Typically used for logging, checkpointing, etc.
    
    # Create a tensor on the designated CUDA device
    tensor = torch.tensor([dist.get_rank()], dtype=torch.float32).to(device)
    print('tensor', tensor, tensor.shape, tensor.device)


    # Prepare list to gather tensors from all processes (on GPU)
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(ddp_world_size)]
    print('gathered_tensors', gathered_tensors, gathered_tensors[0].shape, gathered_tensors[0].device)
    # Perform all_gather operation
    dist.all_gather(gathered_tensors, tensor)
    print('after all gather', gathered_tensors, gathered_tensors[0].shape, gathered_tensors[0].device)
    
    print(f"Process {ddp_rank} gathered tensors: {gathered_tensors}")
    
    # Create another tensor on the CUDA device for all_reduce
    tensor = torch.tensor([1.0], dtype=torch.float32).to(device)
    
    # Perform all_reduce with summation
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    print(f"Process {ddp_rank} has tensor after all_reduce: {tensor}")

if __name__ == "__main__":
    main()