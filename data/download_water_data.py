import zipfile
from os.path import exists

from tqdm import tqdm
import requests

"""
A simple script that downloads the water data from zenodo and extracts it.
"""

def download_water_data(filename, overwrite=False):

    already_downloaded = exists(filename)

    if already_downloaded and not overwrite:
        return
    else:
        response = requests.get("https://zenodo.org/records/7981760/files/water.zip?download=1", stream=True)

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(filename, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Could not download file")
        return


if __name__ == "__main__":
    filename = "water.zip"
    directory_to_extract_in = '.'

    download_water_data(filename)

    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_in)



