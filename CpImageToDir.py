from pathlib import Path
from argparse import ArgumentParser
from shutil import copy

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('bbox', type=str)
    arg_parser.add_argument('sub_dir', type=str)
    args = arg_parser.parse_args()
    bbox = Path(args.bbox)
    sub_dir = Path(args.sub_dir)
    if not bbox.is_file():
        raise FileNotFoundError(str(bbox.absolute()))
    if not sub_dir.is_dir():
        raise NotADirectoryError(str(sub_dir.absolute()))

    with bbox.open('r') as f:
        paths = (
            p.strip().split()[0]
            for p in f.readlines()
        )

    for p in paths:
        image_path = Path(p)
        cp_path = sub_dir / image_path.name
        copy(str(image_path), str(cp_path))
        print('%s -> %s' % (image_path.absolute(), cp_path.absolute()))
