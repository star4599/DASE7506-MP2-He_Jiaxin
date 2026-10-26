"""Verify the frozen submission files and inference assets without prediction."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((ROOT/'submission_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Submission hash mismatch: {name}')
    checkpoint = ROOT/'code/output'
    assets = sum((checkpoint/name).stat().st_size for name in ('model.npz','config.json'))
    if assets != manifest['inference_assets']['uncompressed_bytes'] or assets > 16*2**20:
        raise ValueError('Unexpected inference asset size.')
    with zipfile.ZipFile(ROOT/'artifacts/model_bundle.zip') as archive:
        if sorted(archive.namelist()) != ['config.json','model.npz']:
            raise ValueError('Checkpoint archive must contain only model.npz and config.json.')
        for name in archive.namelist():
            if archive.read(name) != (checkpoint/name).read_bytes():
                raise ValueError(f'Checkpoint archive mismatch: {name}')
    print(json.dumps(dict(status='passed',files_verified=len(manifest['files_sha256']),
        inference_assets_bytes=assets,inference_assets_MiB=assets/2**20,
        matching_checkpoint_archive=True,training_run=False,prediction_run=False),indent=2))


if __name__=='__main__':
    main()
