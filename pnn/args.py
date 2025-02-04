"""
Functions for handling command-line args.
"""
import argparse
from functools import partial

ArgumentParser = partial(argparse.ArgumentParser, formatter_class=argparse.RawDescriptionHelpFormatter)
