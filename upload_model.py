from huggingface_hub import HfApi
import getpass

token = getpass.getpass('Enter your HF token: ')
api = HfApi()
print('Uploading igala_gpt_final.pt...')

api.upload_file(
    path_or_fileobj='outputs/model_checkpoints/igala_gpt_final.pt',
    path_in_repo='outputs/model_checkpoints/igala_gpt_final.pt',
    repo_id='Faruna01/igala-gpt-from-scratch',
    repo_type='space',
    token=token
)
print('Upload complete!')
