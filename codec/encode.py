"""Offline entropy re-encoding of saved HPC symbols, with no optimization."""
from pathlib import Path
import struct

import torch

@torch.no_grad()
def encode(args):
    from hpc_codec import pack, write_rate_report
    from codec.sequence import decode_frames
    from utils.encodings import encoder_factorized
    from utils.networks import NonNNEMGaussian

    pack(args)
    output = Path(args.output)
    for index, model in decode_frames(output):
        if index == 0:
            continue  # Preserve the original complete initial-frame payload.
        directory = output / f'frame{index:06d}'
        latents = torch.cat(model.latent_list, dim=0).contiguous()
        _, low, high = encoder_factorized(latents, model.entropy_model,
                                          str(directory / 'latent.b'))
        with (directory / 'latent_header.b').open('wb') as fout:
            fout.write(struct.pack('>ff', float(low), float(high)))
        for path in sorted((directory / 'nn_bitstream').rglob('*.b')):
            if path.name.endswith(('_header.b', 'cache_ratio.b')):
                continue
            symbols = NonNNEMGaussian.decompress(str(path))
            NonNNEMGaussian.compress(symbols, str(path))
    write_rate_report(output, range(args.end + 1))
    print('Offline symbol entropy re-encoding complete; quantization ranges retained')
