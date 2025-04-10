import os

class S3Syncer:
    def __init__(self):
        pass
    def sync_folder_to_s3(self,folder,aws_bucket_url):
        command = f"aws s3 sync {folder} {aws_bucket_url}"
        os.system(command)
    
    def sync_file_to_s3(self,file,aws_bucket_url):
        command = f"aws s3 cp {file} {aws_bucket_url}"
        os.system(command)