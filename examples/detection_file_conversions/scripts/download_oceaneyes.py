
import requests
import csv

input_csv = 'input.csv'
viame_install = '/home/matt/Dev/viame/build/install'
last_fname = ''

with open( input_csv, newline='' ) as fin:
  spamreader = csv.reader( fin )
  all_files = {}
  for row in spamreader:
    if len( row ) == 0 or row[0][0:5] == "photo":
      continue
    url = row[0]
    fname = row[1]
    if fname in all_files:
      continue
    print( "Downloading " + fname )
    r = requests.get( url, allow_redirects=True )
    open( fname, 'wb' ).write( r.content )
    all_files.add( fname )
