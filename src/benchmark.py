'''Utilities for measuring model inference cost.'''

import time
import numpy as np
import torch


def benchmark_inference(model, batch, device, warmup=10, repeats=50):
    '''Measure forward passes on one fixed, already encoded batch.'''
    if warmup < 0 or repeats < 1:
        raise ValueError('Invalid warmup or repeat count.')

    device = torch.device(device)
    previous_mode = model.training
    model.eval()

    # Prepare inputs before timing; lengths remain on CPU for packing.
    tokens = batch['tokens'].to(device)
    lengths = batch['lengths'].cpu()

    parameter_count = sum(p.numel() for p in model.parameters())
    parameter_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    try:
        with torch.inference_mode():
            for _ in range(warmup):
                output = model(tokens, lengths)
                del output

            synchronize()

            # Subtract existing allocations, including model and inputs.
            if device.type == 'cuda':
                baseline_bytes = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)

            durations_ms = []

            for _ in range(repeats):
                synchronize()
                start = time.perf_counter()

                output = model(tokens, lengths)

                synchronize()
                durations_ms.append((time.perf_counter() - start) * 1000)
                del output

            if device.type == 'cuda':
                extra_peak_mib = (torch.cuda.max_memory_allocated(device) - baseline_bytes) / (1024 ** 2)
            else:
                extra_peak_mib = float('nan')

    finally:
        model.train(previous_mode)

    mean_ms = float(np.mean(durations_ms))
    batch_size = tokens.shape[0]

    return {
        'parameters': parameter_count,
        'parameter_memory_mib': parameter_bytes / (1024 ** 2),
        'batch_size': batch_size,
        'padded_sequence_length': tokens.shape[1],
        'mean_real_length': float(lengths.float().mean()),
        'mean_batch_ms': mean_ms,
        'std_batch_ms': float(np.std(durations_ms, ddof=1)) if repeats > 1 else 0.0,
        'amortized_ms_per_email': mean_ms / batch_size,
        'emails_per_second': batch_size * 1000 / mean_ms,
        'extra_peak_cuda_memory_mib': extra_peak_mib,
        'warmup': warmup,
        'repeats': repeats,
        'device': str(device),
        'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'torch_version': str(torch.__version__),
        'cpu_threads': torch.get_num_threads(),
    }