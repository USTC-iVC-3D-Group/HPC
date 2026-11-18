"""Package trained HPC streams and decode them in a fresh process."""
import argparse
import csv
from pathlib import Path
import shutil

from codec.io import FORMAT, read_json, write_json

def pack(args):
    output = Path(args.output)
    sources = []
    source_root = Path(args.source)
    packaged = (source_root / 'manifest.json').is_file()
    for index in range(args.end + 1):
        iteration = args.initial_iteration if index == 0 else args.iteration
        source = source_root / f'frame{index:06d}'
        if not packaged:
            source = source / f'iteration_{iteration}' / 'bitstreams'
        cfg = source / 'frame_config.json'
        info = read_json(cfg)
        sources.append((index, source, info))
    model = sources[0][2]['model']
    for index, source, info in sources:
        target = output / f'frame{index:06d}'
        target.mkdir(parents=True, exist_ok=True)
        allowed = {'anchor.pkl', 'initial_networks.pt', 'initial_aux.pt',
                   'initial_header.json', 'frame_config.json', 'latent.b', 'latent_header.b'}
        for path in source.iterdir():
            if path.name == 'nn_bitstream':
                shutil.copytree(path, target / path.name, dirs_exist_ok=True)
            elif path.name in allowed or (index == 0 and path.suffix == '.b'):
                shutil.copy2(path, target / path.name)
    manifest = {'format': FORMAT, 'model': model,
                'frames': [{'frame_idx': index} for index, _, _ in sources]}
    write_json(output / 'manifest.json', manifest)
    rows = write_rate_report(output, [index for index, _, _ in sources])
    print(f'Packed {len(rows)} frames; frame files={sum(row["total_bytes"] for row in rows)} bytes; '
          f'manifest={(output / "manifest.json").stat().st_size} bytes')

def write_rate_report(output, indices):
    output = Path(output)
    rows = []
    for index in indices:
        paths = list((output / f'frame{index:06d}').rglob('*'))
        paths = [path for path in paths if path.is_file()]
        latent = sum(path.stat().st_size for path in paths if path.name == 'latent.b')
        network = sum(path.stat().st_size for path in paths
                      if 'nn_bitstream' in path.parts and path.name.endswith('.b')
                      and not path.name.endswith(('_header.b', 'cache_ratio.b')))
        total = sum(path.stat().st_size for path in paths)
        rows.append({'frame_idx': index, 'latent_bytes': latent,
                     'network_payload_bytes': network, 'other_bytes': total - latent - network,
                     'total_bytes': total})
    with (output / 'rate_report.csv').open('w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return rows

def decode(args):
    import torch
    from codec.sequence import decode_frames, representation
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for index, model in decode_frames(args.input):
        frame = output / f'frame{index:06d}'
        frame.mkdir(exist_ok=True)

        model.save_ply(str(frame / 'point_cloud.ply'))
        model.save_mlp_checkpoints(str(frame / 'networks.pt'))
        if args.save_tensors:
            torch.save(representation(model), frame / 'representation.pt')
        print(f'Decoded frame {index}', flush=True)

def encode(args):
    from codec.encode import encode as run
    run(args)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('pack', help='Collect encoded frame files')
    p.add_argument('--source', required=True, help='Sequence training output directory')
    p.add_argument('--output', required=True)
    p.add_argument('--end', type=int, required=True)
    p.add_argument('--initial-iteration', type=int, default=15000)
    p.add_argument('--iteration', type=int, default=500)
    p.set_defaults(func=pack)
    p = commands.add_parser('encode', help='Entropy encode retained symbols')
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--end', type=int, required=True)
    p.add_argument('--initial-iteration', type=int, default=15000)
    p.add_argument('--iteration', type=int, default=500)
    p.set_defaults(func=encode)
    p = commands.add_parser('decode', help='Decode using only the packaged stream')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--save-tensors', action='store_true')
    p.set_defaults(func=decode)
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
