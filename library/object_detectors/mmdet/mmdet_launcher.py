# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import sys
import pickle
import os
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    parser = argparse.ArgumentParser( description='Train a detector' )
    parser.add_argument( 'config', help='train config file path' )
    parser.add_argument('--local_rank', type=int, default=0 )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str( args.local_rank )
    infile=open( args.config,'rb' );
    trainer=pickle.load( infile );
    trainer.internal_update();

if __name__ == "__main__":
    main()
