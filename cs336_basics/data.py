import numpy.typing as npt
import numpy as np
import torch

def get_batch(
    dataset: npt.NDArray,
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    
    if dataset.ndim != 1:
        raise ValueError(f"dataset needs to be 1D, but got {dataset.ndim}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must > 0, but got {batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must > 0, but got {context_length}")

    n = dataset.size
    if n < context_length + 1:
        raise ValueError(f"dataset must be least context_length+1 tokens, got{n}, context_legnth={context_length}")

    # sample input
    max_start = n - context_length - 1
    starts = np.random.randint(0, max_start+1, size=batch_size)

    # numpy advanced indexing/broadcasting to construct index matrix (batch_size, context_legnth+1)
    offsets = np.arange(context_length+1)
    indices = starts.reshape(batch_size, 1) + offsets

    block = dataset[indices]

    # split
    input_np = block[:, :-1]
    output_np = block[:, 1:]

    inputs = torch.from_numpy(input_np).to(device=device, dtype=torch.long)
    outputs = torch.from_numpy(output_np).to(device=device, dtype=torch.long)

    return (inputs, outputs)
