from mfit.mfit1 import main as m1
from mfit.mfit2 import main as m2
import argparse
import sys

def mfit1():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs='*', type=str, help='input files')
    m1.main()
    
def mfit2():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs='*', type=str, help='input files')
    m2.main()
    
    