"""Deprecation guard for old model and script versions.

Usage (add to top of deprecated file, after docstring):

    from gnn.deprecation import deprecated
    deprecated(__file__, "Use atlas_transformer_v6.py instead.")
"""

import sys
import os


def deprecated(filepath, suggestion):
    """Print deprecation warning and exit unless --force is passed.

    Args:
        filepath: __file__ of the deprecated module
        suggestion: message pointing to the replacement
    """
    name = os.path.basename(filepath)
    msg = (
        f"\n  DEPRECATED: {name}\n"
        f"  {suggestion}\n"
        f"  Pass --force to run anyway.\n"
    )

    # If imported (not run directly), just warn
    if not _is_main(filepath):
        import warnings
        warnings.warn(f"DEPRECATED: {name} — {suggestion}", DeprecationWarning, stacklevel=3)
        return

    # If run as script, block unless --force
    if "--force" in sys.argv:
        sys.argv.remove("--force")
        print(f"  WARNING: {name} is deprecated. {suggestion}", file=sys.stderr)
        return

    print(msg, file=sys.stderr)
    sys.exit(1)


def _is_main(filepath):
    """Check if this module is being run as __main__."""
    import __main__
    main_file = getattr(__main__, "__file__", None)
    if main_file is None:
        return False
    return os.path.realpath(main_file) == os.path.realpath(filepath)
