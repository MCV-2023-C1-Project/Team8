# C1. Content Based Image Retrieval 

## Folder structure
The code and data should be structured as follows:

        .
        ├── C1_w2_team8.ipynb     # Source code
        ├── data                  
        │   ├── BBDD              # Paintings dataset
        │   ├── qsd1_w3           # Query set 1 for development
        │   └── qsd2_w3           # Query set 2 for development
        │   └── qst1_w3           # Query set 1 for test
        │   └── qst2_w3           # Query set 2 for test
        └── ...

## Requirements

We use standard Machine Learning / Computer Vision python packages. In case any of the packages are missing, we provide some commands for quick installation:

- OpenCV / cv2: $pip install opencv-python
- Tqdm: $pip installl tqdm
- Pickle: $pip install pickle5

They all can be installed with:
```console
$pip install -r requirements.txt
```
Regarding the Python version, Python >= 3.8 is needed.

## Running the code
Since the code is in _.ipynb_ format, it is required to have _Jupyter Notebook_ or any other program/text editor that can run this kind of file. We recommend using Visual Studio Code with the _Jupyter_ extension.

To replicate the submitted results and ensure that everything works as expected, simply use the __Run all__ button (or run the code blocks from top to bottom).
