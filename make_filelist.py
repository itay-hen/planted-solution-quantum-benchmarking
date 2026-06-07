#!/usr/bin/env python3
"""
make_filelist.py -- emit a list of shell commands for batch processing.

Replacement for the original ``filelistgenerator.py``.  Improvements: the
target script and the directory are arguments (not hard-coded), filenames
are shell-quoted, and it scans the requested directory rather than only
the current one.

Usage:
    python make_filelist.py --dir suite --script spectrum_single.py > jobs.sh
    python make_filelist.py --dir suite --script mingap_single.py --python python3 > jobs.sh
"""

import argparse
import glob
import os
import shlex
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description="emit batch commands over *_instance.json files")
    p.add_argument("--dir", default=".", help="directory to scan")
    p.add_argument("--script", required=True, help="script to invoke per instance")
    p.add_argument("--python", default=sys.executable, help="python interpreter")
    p.add_argument("--by-basename", action="store_true",
                   help="pass BASENAME instead of the instance path")
    args = p.parse_args(argv)

    pattern = os.path.join(args.dir, "*_instance.json")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.stderr.write(f"no *_instance.json files found in {args.dir!r}\n")
        return 1
    for fpath in files:
        if args.by_basename:
            arg = fpath[: -len("_instance.json")]
        else:
            arg = fpath
        print(f"{shlex.quote(args.python)} {shlex.quote(args.script)} {shlex.quote(arg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
