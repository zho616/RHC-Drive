"""Export the installed MetaDrive source/assets for exact reproduction on another host."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def source_hashes(package):
    return {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob('*.py')) if '__pycache__' not in p.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New .tar.gz archive; never uploaded by this script')
    parser.add_argument('--check', type=Path, help='Verify installed simulator against a JSON source manifest')
    args = parser.parse_args()
    from metadrive.envs import metadrive_env
    package = Path(metadrive_env.__file__).resolve().parents[1]
    hashes = source_hashes(package)
    if args.check:
        expected = json.loads(args.check.read_text())['source_sha256']
        missing = sorted(set(expected) - set(hashes))
        changed = sorted(key for key in expected.keys() & hashes.keys() if expected[key] != hashes[key])
        print(json.dumps(dict(missing=missing, changed=changed, matching=len(expected)-len(missing)-len(changed)), indent=2))
        if missing or changed:
            raise SystemExit('Simulator differs from the local training runtime; transfer/install the exported source/assets')
    if args.output:
        output = args.output.expanduser().resolve()
        if output.exists():
            parser.error('Use a fresh output archive')
        if not (package / 'assets').is_dir():
            parser.error('Installed runtime assets are missing')
        output.parent.mkdir(parents=True, exist_ok=True)
        def include(info):
            if '__pycache__' in Path(info.name).parts or info.name.endswith(('.pyc', '.pyo')):
                return None
            return info
        with tarfile.open(output, 'w:gz') as archive:
            archive.add(package, arcname='metadrive-runtime/metadrive', filter=include)
            for name in ('setup.py', 'MANIFEST.in', 'README.md', 'LICENSE.txt'):
                archive.add(package.parent / name, arcname='metadrive-runtime/' + name)
        manifest = dict(source_sha256=hashes, archive=output.name,
                        archive_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), bytes=output.stat().st_size)
        output.with_suffix(output.suffix + '.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({k:v for k,v in manifest.items() if k!='source_sha256'},indent=2))


if __name__ == '__main__':
    main()
