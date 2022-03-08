import json
import pickle
from pathlib import Path
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('json_file', help='json file', type=str)
    args = parser.parse_args()
    json_file_path = Path(args.json_file)

    if not json_file_path.is_file():
        raise FileNotFoundError(str(json_file_path.absolute()))

    pickle_file_path = (json_file_path.parent / json_file_path.stem).with_suffix('.pickle')
    pickle_file_path.touch(exist_ok=True)

    with json_file_path.open('r') as f:
        json_dic = json.load(f)
    with pickle_file_path.open('wb') as f:
        pickle.dump(json_dic, f)
