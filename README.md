# MMRT-Fitting
GUI applications for fitting of Macromolecular Rate Theory (MMRT) equations.

Details about MMRT can be found here:

https://www.biorxiv.org/content/10.1101/2023.07.06.548038v1

https://onlinelibrary.wiley.com/doi/full/10.1111/gcb.12596 


## MFit1
Fits only MMRT 1.0 (Constant activation heat capacity) and MMRT 1.5 (Linear activation heat capacity)

## Mfit2
Fits only MMRT 2.0 (Two state, sigmoidal activation heat capacity)

## Running MFit
Download this repository or clone using git then install using pip

```bash
git clone https://github.com/CarlinHamill/mfit/
cd mfit
pip install .
```

GUI can then be run with the commands 'mfit1' or 'mfit2'

MFit1:
```bash
mfit1
```

MFit2:
```bash
mfit2
```

## Dependencies
- numpy
- matplotlib
- pandas
- openpyxl
- scipy
- lmfit
- PyQt6
- numba