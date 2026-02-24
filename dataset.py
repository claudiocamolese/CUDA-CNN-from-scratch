import shutil
import os
import gzip
import urllib.request
import argparse

def main(args):
    """Download dataset specified
    """

    base_dir = os.path.join("./datasets", args.data)
    os.makedirs(base_dir, exist_ok=True)
    
    if args.data == "mnist":
        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    elif args.data == "fashion":
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        
    files = [
        ("train-images-idx3-ubyte.gz", "train-images.idx3-ubyte"),
        ("train-labels-idx1-ubyte.gz", "train-labels.idx1-ubyte"),
        ("t10k-images-idx3-ubyte.gz", "t10k-images.idx3-ubyte"),
        ("t10k-labels-idx1-ubyte.gz", "t10k-labels.idx1-ubyte")
    ]

    for gz_file, filename in files:
        url = base_url + gz_file
        gz_path = os.path.join(base_dir, filename + ".gz")
        file_path = os.path.join(base_dir, filename)

        print(f"Downloading {url} to {gz_path} ...")
        urllib.request.urlretrieve(url, gz_path)
        
        print(f"Extracting {gz_path} to {file_path} ...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"Removing {gz_path} ...")
        os.remove(gz_path)
        print(f"{file_path} is ready.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["mnist", "fashion"], required=True, 
                        help="Select the dataset to download")
    args = parser.parse_args()
    main(args)
