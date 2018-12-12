#!/usr/bin/python

import os
import sys
import re
import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_roc( fn ):

  x_fa = np.array( [] )
  y_pd = np.array( [] )

  with open(fn) as f:
    while (1):
      raw_line = f.readline()
      if not raw_line:
        break
      fields = raw_line.split()
      x_fa = np.append( x_fa, float( fields[47] ))
      y_pd = np.append( y_pd, float( fields[7] ))

  return (x_fa, y_pd)

if __name__ == "__main__":
  parser = argparse.ArgumentParser( description = 'Generate detection scores and ROC plots' )

  parser.add_argument( '-rangey', metavar='rangey', nargs='?', default='0:1',
             help='ymin:ymax (quote w/ spc for negative, i.e. " -0.1:5")' )
  parser.add_argument( '-rangex', metavar='rangex', nargs='?',
             help='xmin:xmax (quote w/ spc for negative, i.e. " -0.1:5")' )
  parser.add_argument( '-autoscale', action='store_true',
             help='Ignore -rangex -rangey and autoscale both axes of the plot.' )
  parser.add_argument( '-logx', action='store_true',
             help='Use logscale for x' )
  parser.add_argument( '-xlabel', nargs='?', default='Detection FA count',
             help='title for x axis' )
  parser.add_argument( '-ylabel', nargs='?', default='Detection PD',
             help='title for y axis' )
  parser.add_argument( '-title', nargs='?',
             help='title for plot' )
  parser.add_argument( '-lw', nargs='?', type=float, default=3,
             help='line width' )
  parser.add_argument( '-key', nargs='?', default=None,
             help='comma-separated set of strings labeling each line in order read' )
  parser.add_argument( '-keyloc', nargs='?', default='best',
             help='Key location ("upper left", "lower right", etc; help for list)' )
  parser.add_argument( '-nokey', action='store_true',
             help='Set to suppress plot legend' )
  parser.add_argument( '-writeimage', default=None,
             help='Provide a filename of the image to write instead of displaying a window.' )
  parser.add_argument( 'rocfiles', metavar='ROC', nargs=argparse.REMAINDER,
             help='A score_events .roc file (from --roc-dump argument)' )
  parser.add_argument("--debug", dest="debug", action="store_true",
             help="Run with debugger attached to process")

  args = parser.parse_args()

  fig = plt.figure()
  xscale_arg = 'log' if args.logx else 'linear'
  rocplot = plt.subplot(1, 1, 1, xscale=xscale_arg)
  rocplot.set_title( args.title ) if args.title else None
  plt.xlabel( args.xlabel )
  plt.ylabel( args.ylabel )
  plt.xticks()
  plt.yticks()

  user_titles = args.key.split(',') if args.key else None
  i = 0
  for fn in args.rocfiles:
    (x,y) = load_roc( fn )
    t = user_titles[i] if user_titles and i < len(user_titles) else fn
    sys.stderr.write("Info: %d: loading %s as '%s'...\n" % (i, fn, t) )
    rocplot.plot( x, y, linewidth=args.lw, label=t )
    i += 1

  if args.autoscale:
    rocplot.autoscale()
  else:
    tmp = args.rangey.split(':')
    if len(tmp) != 2:
      sys.stderr.write('Error: rangey option must be two floats separated by a colon, e.g. 0.2:0.7\n');
      sys.exit(1)
    (ymin, ymax) = (float(tmp[0]), float(tmp[1]))
    rocplot.set_ylim(ymin,ymax)

    if args.rangex:
      tmp = args.rangex.split(':')
      if len(tmp) != 2:
        sys.stderr.write('Error: rangex option must be two floats separated by a colon, e.g. 0.2:0.7\n');
        sys.exit(1)
      (xmin, xmax) = (float(tmp[0]), float(tmp[1]))
      rocplot.set_xlim(xmin,xmax)

  if not args.nokey:
    plt.legend( loc=args.keyloc )

  if args.writeimage:
    plt.savefig(args.writeimage)
  else:
    plt.show()

