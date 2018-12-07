import os
from PIL import Image
import sys
import os
import subprocess
import time
import sys
import shlex
from os.path import expanduser
from textwrap import dedent
import getpass

# http://code.activestate.com/recipes/578933-pasting-python-data-into-a-spread-sheet/
def _buildstr(D, transpose=False, replace=None):
    """Construct a string suitable for a spreadsheet.

    D: scalar, 1d or 2d sequence
        For example a list or a list of lists.

    transpose: Bool
        Transpose the data if True.

    replace: tuple or None
        If tuple, it is two strings to pass to the replace
        method. ('toreplace', 'replaceby')

    """

    try:
        D[0]
    except (TypeError, IndexError):
        D = [D]
    try:
        D[0][0]
    except (TypeError, IndexError):
        D = [D]

    if transpose:
        D = zip(*D)

    if not replace:
        # changed here a little bit from the source.
        rows = ['\t'.join(['%.2f' % (v) for v in row]) for row in D]
    else:
        rows = ['\t'.join([str(v).replace(*replace)
                           for v in row]) for row in D]
    S = '\n'.join(rows)
    return S


def save_images( visuals, image_path, results_dir ):

    assert visuals.shape[0] == len(image_path)
    for i in range(len(image_path)):
        img = visuals[i]
        img_name = os.path.basename(image_path[i])
        pil_image = Image.fromarray(img)
        pil_image.save(os.path.join(results_dir , img_name))



def save_images_gen( visuals, image_path, results_dir ):

    assert visuals.shape[0] == len(image_path)
    for i in range(len(image_path)):
        img = visuals[i]
        img_loc = os.path.join(results_dir , image_path[i])
        img_dir   = os.path.dirname(img_loc)

        # print results_dir , img_loc
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        pil_image = Image.fromarray(img)
        pil_image.save(img_loc)



def save_params( file_path , args ):
    args = vars(args)
    with open(file_path, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')


# This is the class for writing both in screen and logfile
# https://stackoverflow.com/questions/17866724/python-logging-print-statements-while-having-them-print-to-stdout
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()


#Some utility codes:https://stackoverflow.com/questions/1556348/python-run-a-process-with-timeout-and-capture-stdout-stderr-and-exit-status
class Timeout(Exception):
    pass

def run(command, timeout=10):
    proc = subprocess.Popen(command, bufsize=0, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    poll_seconds = .250
    deadline = time.time() + timeout
    while time.time() < deadline and proc.poll() == None:
        time.sleep(poll_seconds)

    if proc.poll() == None:
        if float(sys.version[:3]) >= 2.6:
            proc.terminate()
        raise Timeout()

    stdout, stderr = proc.communicate()
    return stdout, stderr, proc.returncode