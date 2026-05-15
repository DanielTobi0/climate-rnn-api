"""
Upload Model to HuggingFace Hub

Uploads prepared model artifacts to a HuggingFace repository.
"""

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, login


def main():
    project_root = Path(__file__).parent.parent
    artifacts_dir = project_root / 'artifacts'

    print('Uploading model to HuggingFace Hub...')
    print('=' * 60)

    # Configuration
    repo_id = 'DanielTobi0/climate-rnn-model'
    print(f'Repository: {repo_id}')
    print(f'Artifacts directory: {artifacts_dir}')

    if not artifacts_dir.exists():
        print(f'\nArtifacts directory not found: {artifacts_dir}')
        print('Run prepare_hf_model.py first!')
        return 1

    required_files = ['pytorch_model.bin', 'scaler.pkl', 'config.json', 'README.md']
    missing = [f for f in required_files if not (artifacts_dir / f).exists()]

    if missing:
        print(f'\nMissing required files: {", ".join(missing)}')
        print('Run prepare_hf_model.py first!')
        return 1

    print('\n✓ All required files present:')
    for file in required_files:
        file_path = artifacts_dir / file
        print(f'  - {file} ({file_path.stat().st_size:,} bytes)')

    token = os.environ.get('HF_TOKEN')
    if not token:
        print('\nHF_TOKEN environment variable not set!')
        print('Get your token from: https://huggingface.co/settings/tokens')
        print('Then set it: export HF_TOKEN=your_token_here')
        return 1

    try:
        print('\nLogging in to HuggingFace...')
        login(token=token, add_to_git_credential=False)
        print('✓ Logged in successfully')

        api = HfApi()

        print(f'\nCreating repository: {repo_id}')
        create_repo(repo_id, exist_ok=True, repo_type='model', private=True, token=token)
        print('✓ Repository created/verified (private)')

        print('\nUploading files...')
        api.upload_folder(
            folder_path=str(artifacts_dir),
            repo_id=repo_id,
            repo_type='model',
            token=token,
            commit_message='Upload climate forecasting RNN model',
        )

        print('\nUpload complete!')
        print('=' * 60)
        print(f'\nModel available at: https://huggingface.co/{repo_id}')
        print('\nNext steps:')
        print('1. Set HF_TOKEN in your .env file or Render dashboard')
        print('2. Update pyproject.toml with production dependencies')
        print('3. Test the API locally')
        print('4. Deploy to Render')

    except Exception as e:
        print(f'\nUpload failed: {e}')
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
