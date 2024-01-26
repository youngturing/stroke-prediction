import subprocess

from zipfile import ZipFile


def download_data_zip() -> None:
    command = 'kaggle datasets download -d fedesoriano/stroke-prediction-dataset'
    subprocess.run(command.split(' '))


def extract_data_to_csv(path: str) -> None:
    """
    Parameters
    ----------
        path: downloaded filename with healthcare data.
    """
    with ZipFile(path, 'r') as zip_obj:
        zip_obj.extractall()


def main(path):
    download_data_zip()
    extract_data_to_csv(path)


if __name__ == '__main__':
    path = './stroke-prediction-dataset.zip'
    main(path)
