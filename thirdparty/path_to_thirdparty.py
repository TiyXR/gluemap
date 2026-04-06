import sys
import os.path as path

HERE_PATH = path.normpath(path.dirname(__file__))

SUBMODULES = {
    'pi3': path.join('pi3', 'pi3', 'models'),
    'salad': path.join('salad', 'vpr_model.py'),
    'doppelgangers-plusplus': path.join('doppelgangers-plusplus', 'mast3r'),
    'vggt': path.join('vggt', 'vggt', 'models'),
    'mapanything': path.join('mapanything', 'mapanything', 'models'),
}

for name, check_path in SUBMODULES.items():
    repo_path = path.normpath(path.join(HERE_PATH, name))
    full_check = path.join(HERE_PATH, check_path)
    if path.exists(full_check):
        sys.path.insert(0, repo_path)
    else:
        raise ImportError(
            f"{name} is not initialized, could not find: {full_check}.\n "
            "Did you forget to run 'git submodule update --init --recursive' ?"
        )
