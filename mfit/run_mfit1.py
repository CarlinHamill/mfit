from mfit.mfit1 import main as m1
import argparse
import sys
import pyi_splash

def mfit1():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs='*', type=str, help='input files')
    m1.main()

if __name__ == '__main__':
    pyi_splash.update_text('UI Loaded ...')
    pyi_splash.close()
    mfit1()