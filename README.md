# Automatic generation of areas of interest (AOI)
Python scripts developed for the analysis made for the manuscript:


Martyna A. Galazka, Lena Wallin, Max Thorsson, Christopher Gillberg, Eva Billstedt, Nouchine Hadjikhani & Jakob Åsberg Johnels (2023) Self-reported eye contact sensitivity and face processing in chromosome 22q11.2 deletion syndrome, Journal of Clinical and Experimental Neuropsychology, DOI: [10.1080/13803395.2023.2259043](https://doi.org/10.1080/13803395.2023.2259043)


## The responsory features:
* Automatic generation of AOIs for facial images 
* Classification of gaze within AOIs

## How to use
Tested with Python 3.8.
### Usage:
* "python3 detect.py" to detect and store facial AOIs
* "python3 classify.py" to get classification of gaze to AOIs

More details in files.

## Dependencies
* OpenCV
* Dlib
* imutils
* Pandas
* Numba
* tqdm
* cuda

### Detection models:
* 'mmod_human_face_detector.dat' (http://dlib.net/files/mmod_human_face_detector.dat.bz2)
* 'shape_predictor_68_face_landmarks.dat' (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)


## Citation
On top of the conditions stipulated in the software license (see the LICENSE file), you are kindly asked to cite our manuscript if you make use of this code in any academic context.

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
