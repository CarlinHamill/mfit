from mfit.mfit2 import main as m2
import argparse
import sys
import pyi_splash

def mfit2():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs='*', type=str, help='input files')
    m2.main()

if __name__ == '__main__':
    pyi_splash.update_text('UI Loaded ...')
    pyi_splash.close()
    mfit2()