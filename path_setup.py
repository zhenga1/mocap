"""
Ensure project root is on sys.path so amc_parser and other root-level modules
are importable from any script (e.g. in scripts/, demos/, or project root).

Usage from a script that lives under the project:
  import path_setup
  path_setup.ensure_project_root(__file__)
  import amc_parser

Or add the root yourself before any other imports:
  import os, sys
  _r = os.path.dirname(os.path.abspath(__file__))
  while _r and not os.path.isfile(os.path.join(_r, "amc_parser.py")):
      _r = os.path.dirname(_r)
  if _r and _r not in sys.path:
      sys.path.insert(0, _r)
  import amc_parser
"""
import os
import sys


def ensure_project_root(caller_file=None):
    """
    Ensure the directory containing amc_parser.py is on sys.path.
    Call with __file__ from the script that needs to import amc_parser.
    If caller_file is None, uses the caller's __file__ via inspect.
    """
    if caller_file is None:
        try:
            import inspect
            frame = inspect.stack()[1]
            caller_file = frame.filename
        except Exception:
            caller_file = ""
    if not caller_file:
        # Fallback: this module's directory is project root
        _root = os.path.dirname(os.path.abspath(__file__))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        return _root
    start = os.path.dirname(os.path.abspath(caller_file))
    root = start
    while root and not os.path.isfile(os.path.join(root, "amc_parser.py")):
        root = os.path.dirname(root)
    if root and root not in sys.path:
        sys.path.insert(0, root)
    return root
