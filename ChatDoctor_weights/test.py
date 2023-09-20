import gdown

from google_drive_downloader import GoogleDriveDownloader as gdd

#gdd.download_file_from_google_drive(
#    file_id='11-qPzz9ZdHD6pc47wBSOUSU61MaDPyRh',
#    dest_path='./test.zip',
#    unzip=True)

google_path = 'https://drive.google.com/uc?id='
file_id = '1_GsLERArX4zHbm-nOEf8a7F8K5kdsTbW'
output_name = 'pytorch_model-00003-of-00003.bin'
gdown.download(google_path+file_id,output_name,quiet=False)
