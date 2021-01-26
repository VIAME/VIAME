 #!/usr/bin/env python

import sys
import os
import shutil
import argparse
import glob

# Main Function
if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="Check GPU properties of the system",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--use-pytorch", dest="use_pytorch", action="store_true",
                        help="Use pytorch for checking gpu properties")

    args = parser.parse_args()
    use_pytorch = True

    print( "" )
    print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" )
    print( "~~~~~~~~~~VIAME GPU CHECK UTILITY~~~~~~~~~~~~~" )
    print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" )
    print( "" )

    if use_pytorch:

        import torch
        if not torch.cuda.is_available():
            print( "Unable to access CUDA on this system" )
            sys.exit( 0 )

        gpu_count = torch.cuda.device_count()
        print( "Usable devices: " + str( gpu_count ) + "\n" )
        for i in range( gpu_count ):
            gpu_mem = torch.cuda.get_device_properties( i ).total_memory
            print( "Device #1, usable mem: " + str( gpu_mem ) )

    print( "\nExiting\n" )
