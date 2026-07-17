"""
Command-line interface for plGeoAdaptels.

Usage:
    python -m plgeoadaptels -i input.tif -o output.tif -t 60.0
    plgeoadaptels -i band1.tif -i band2.tif -o result.tif -t 40.0
"""

import argparse
import sys

from .adaptels import create_adaptels


def main():
    parser = argparse.ArgumentParser(
        prog='plgeoadaptels',
        description='Scale-Adaptive Superpixels (Adaptels) for geospatial data'
    )
    
    parser.add_argument('-i', '--input', action='append', required=True,
                        help='Input GeoTIFF file(s). Can be specified multiple times.')
    parser.add_argument('-o', '--output', required=True,
                        help='Output GeoTIFF file with adaptel labels.')
    parser.add_argument('-t', '--threshold', type=float, default=60.0,
                        help='Threshold value (default: 60.0)')
    parser.add_argument('-d', '--distance', default='minkowski',
                        choices=['minkowski', 'cosine', 'angular'],
                        help='Distance measure (default: minkowski)')
    parser.add_argument('-p', '--minkowski-p', type=float, default=2.0,
                        help='Minkowski distance parameter (default: 2.0)')
    parser.add_argument('-8', '--queen-topology', action='store_true',
                        help='Use 8-connectivity (default: 4-connectivity)')
    parser.add_argument('-n', '--normalize', action='store_true',
                        help='Normalize input to [0,1]')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Quiet mode')
    
    args = parser.parse_args()
    
    labels, n_adaptels = create_adaptels(
        input_files=args.input,
        output_file=args.output,
        threshold=args.threshold,
        distance=args.distance,
        minkowski_p=args.minkowski_p,
        queen_topology=args.queen_topology,
        normalize=args.normalize,
        quiet=args.quiet,
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
