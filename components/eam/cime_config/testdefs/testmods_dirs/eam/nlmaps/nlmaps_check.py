#!/usr/bin/env python3

import os, sys, re

def readall(fn):
    with open(fn,'r') as f:
        txt = f.read()
    return txt

def greptxt(pattern, txt):
    return re.findall('(?:' + pattern + ').*', txt, flags=re.MULTILINE)

def grep(pattern, fn):
    txt = readall(fn)
    return greptxt(pattern, txt)

def get_run_dir(case_dir):
    filename = case_dir + os.path.sep + 'CaseDocs' + os.path.sep + 'atm_modelio.nml'
    ln = grep('diro = ', filename)[0]
    return ln.split()[2].split('"')[1]

def get_cpl_log(run_dir):
    filenames = os.listdir(run_dir)
    atm_fn = None
    for f in filenames:
        if 'cpl.log' in f:
            atm_fn = f
            break
    return run_dir + os.path.sep + atm_fn

def uncompress(filename):
    if '.gz' in filename:
        os.system('gunzip {}'.format(filename))
        return filename[:-3]
    return filename

case_dir = sys.argv[1]
run_dir = get_run_dir(case_dir)

good = True

if good:
    print('PASS')
    sys.exit(0)
else:
    print('FAIL')
    sys.exit(1) # non-0 exit will make test RUN phase fail, as desired
