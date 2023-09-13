"""Dataset class for pipeline."""
import os
import shutil


def pars_data():
    """Unzip data."""
    filename = './archive.zip'
    extract_dir = './data'
    archive_format = 'zip'

    shutil.unpack_archive(filename, extract_dir, archive_format)

    for name_file in os.listdir('./data/planet/planet'):
        try:
            shutil.copytree(
                './data/planet/planet/{name_file}'.format(name_file=name_file),
                './data/{name_file}'.format(name_file=name_file),
            )
        except Exception:
            shutil.copy(
                './data/planet/planet/{name_file}'.format(name_file=name_file),
                './data',
            )

    shutil.rmtree('./data/planet')
    os.remove('./archive.zip')


if __name__ == '__main__':
    pars_data()
