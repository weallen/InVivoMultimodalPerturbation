
import argparse
from pftools.pipeline.export.zarr_export import convert_tiffs_to_zarr, identify_position_file

def main():
    parser = argparse.ArgumentParser(description='Convert dataset to zarr format')
    parser.add_argument('input', help='Path to input dataset directory with TIFF files')
    parser.add_argument('output', help='Path to output zarr')
    parser.add_argument('--pos_file', default=None, type=str, help='Path to position file. If not specified will use pos_tiled.txt or tiled_pos.txt')
    parser.add_argument('--no_compression', action=argparse.BooleanOptionalAction, help='Path to position file')

    args = parser.parse_args()
    if args.pos_file is None:
        args.pos_file = identify_position_file(args.input)
    if args.no_compression:
        convert_tiffs_to_zarr(args.input, args.output, args.pos_file, use_compression=False)
    else:
        convert_tiffs_to_zarr(args.input, args.output, args.pos_file, use_compression=True)

if __name__ == '__main__':
    main()
