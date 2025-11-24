import os
import zipfile 

def ensure_folder(folder: str) -> None:
    if not os.path.exists(folder):
        os.makedirs(folder)

def extract(filename: str, folder: str) -> None:
    print("Extracting {}...".format(filename))
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall(folder)
    zip_ref.close()


if __name__ == "__main__":
    # Make sure to download the following files into data/ICDAR2015/ before running this script
    # or change the paths into whatever you have
    training_data_path = "data/ICDAR2015/Training/"
    test_data_path = "data/ICDAR2015/Testing/"

    ensure_folder(training_data_path)
    extract("data/ICDAR2015/ch4_training_images.zip", training_data_path)
    extract("data/ICDAR2015/ch4_training_localization_transcription_gt.zip", training_data_path)

    ensure_folder(test_data_path)
    extract("data/ICDAR2015/ch4_test_images.zip", test_data_path)
    extract("data/ICDAR2015/Challenge4_Test_Task1_GT.zip", test_data_path)