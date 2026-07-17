"""Allow running as: python -m plgeoadaptels"""
import sys

from .cli import main

# main()'s return value is the exit code; without sys.exit() a failure would
# still report success, and any script or CI step checking $? would miss it.
sys.exit(main())
