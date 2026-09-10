#!/usr/bin/python

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Plot evaluation results from VIAME model evaluator.

This script reads the CSV/JSON output from the evaluate_models C++ class
and generates publication-quality plots including:
- Precision-Recall curves (overall and per-class)
- Confusion matrices (raw and normalized)
- ROC curves
- IoU distribution histograms
- Track length histograms
- Track purity/continuity histograms
- Summary metric bar charts (MOT, HOTA, detection metrics)

Usage:
    viame plot eval -input <input_dir_or_json> [-output <output_dir>]

Example:
    viame plot eval -input ./eval_output -output ./plots
    viame plot eval -input ./eval_data.json -output ./plots
    viame plot detections fish,shark 0.5 30 -input ./detections -output ./plots
"""

import argparse
import warnings
import datetime
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ----------------- GLOBAL VARIABLES AND PROPERTIES --------------------

DEFAULT_DPI = 150
DEFAULT_FIGSIZE = (8, 6)


# -------------------- GENERIC UTILITY FUNCTIONS -----------------------


# =============================================================================
# Detection count plots (viame plot detections)
# =============================================================================
def create_dir( dirname ):
  if not os.path.exists( dirname ):
    os.makedirs( dirname )

unordered_ext = ".time_ordered.csv"
ranked_ext = ".max_ordered.csv"

def detection_plot( input_directory, output_directory, objects, threshold, frame_rate,
                    smooth=1, ext=".csv", net_name="all_fish", top_category_only=False ):

  def format_x( x, pos=0, show_ms=False ):
    t = datetime.timedelta( seconds = x )
    split_str = str( t ).split(".")
    if show_ms:
      return split_str[0] + ( "." + split_str[1][0] if len( split_str ) > 1 else ".0" )
    else:
      return split_str[0]

  if not np.isfinite(frame_rate) or frame_rate <= 0 or smooth < 1:
    raise ValueError("Frame rate must be finite and positive, and smoothing at least 1")

  if net_name in objects:
    print( "Plotting error, net category id can't be in object list" )
    sys.exit(0)
  else:
    objects = [ net_name ] + objects

  if input_directory == output_directory:
    print( "Plotting error, input and output directories must be different" )
    sys.exit(0)

  video_max_counts = dict()

  for obj in objects:
    video_max_counts[obj] = []

  create_dir( output_directory )

  for filename in os.listdir( input_directory ):

    if not filename.endswith( ext ):
      continue

    filebase = filename.replace( ext, "" )

    # Parse input computed detections file
    video_objects = dict()

    for obj in objects:
      video_objects[obj] = dict()

    with open( os.path.join( input_directory, filename ), "r" ) as f:
      for line in f:
        line = line.rstrip()
        if line and line[0] != "#":
          columns = line.split(",")
          frame_id = int( columns[2] )
          for obj in objects:
            if frame_id not in video_objects[obj]:
              video_objects[obj][frame_id] = 0

          detection_columns = columns[9:11] if top_category_only else columns[9:]
          name = None
          is_first = True
          for column in detection_columns:
            if column.strip().startswith("("):
              break
            if name is not None:
              if name in objects:
                value = float(column)
                if value >= threshold:
                  video_objects[name][frame_id] += 1
                  if is_first:
                    video_objects[net_name][frame_id] += 1
                    is_first = False
              name = None
            else:
              name = column

    # Apply smoothing factor
    for obj in objects:
      smoothed_video_frames = dict()
      for frame_id in sorted( video_objects[obj] ):
        lower_bound = frame_id - smooth // 2
        upper_bound = lower_bound + smooth

        max_count = video_objects[obj][frame_id]
        for i in range( lower_bound, upper_bound ):
          try:
            val = video_objects[obj][i]
            if val > max_count:
              max_count = val
          except KeyError:
            pass

        smoothed_video_frames[frame_id] = max_count

      video_objects[obj] = smoothed_video_frames

    # Write out video specific items alongside aggregrate plot
    video_subdir = os.path.join( output_directory, filebase )
    create_dir( video_subdir )

    agr_fig, agr_ax = plt.subplots()
    agr_plot_title = "Aggregate - " + filename
    agr_ax.xaxis.set_major_formatter( matplotlib.ticker.FuncFormatter( format_x ) )
    agr_ax.xaxis.set_major_locator( matplotlib.ticker.MaxNLocator( integer=True ) )
    agr_ax.set( xlabel="Time", ylabel="Object Count", title=agr_plot_title )
    agr_ax.grid()
    agr_max_y = 0

    for obj in objects:
      sorted_frames = list()
      with open( os.path.join( video_subdir, obj + unordered_ext ), "w" ) as of:
        of.write( "# video_id, time_id, frame_id, detection_count\n" )

        times = list()
        object_counts = list()

        for frame_id in sorted( video_objects[obj] ):
          frame_time = frame_id / frame_rate
          times.append( frame_time )
          object_counts.append( video_objects[obj][frame_id] )

          of.write( filename + "," )
          of.write( format_x(frame_time, show_ms=True) + "," )
          of.write( str(frame_id) + "," )
          of.write( str(video_objects[obj][frame_id]) + "\n" )

          sorted_frames.append( (filename, frame_id, video_objects[obj][frame_id]) )

        x = np.array( times )
        y = np.array( object_counts )

        plot_title = obj + " - " + filename

        fig, ax = plt.subplots()
        ax.xaxis.set_major_formatter( matplotlib.ticker.FuncFormatter(format_x) )
        ax.xaxis.set_major_locator( matplotlib.ticker.MaxNLocator(integer=True) )
        plt.locator_params( axis='x', nbins=6 )
        ax.set( xlabel="Time", ylabel="Object Count", title=plot_title )
        ax.grid()

        ax.plot( x, y )
        ax.set_ylim( ymin = 0 )
        ax.set_xlim( xmin = 0 )
        if np.size( y ) > 0 and np.max( y ) < 5:
          ax.set_ylim( ymax = 5 )
        ax.locator_params( axis='x', nbins = 7 )
        fig.savefig( os.path.join( video_subdir, filename + "." + obj + ".png" ) )

        plt.close(fig)
        agr_ax.plot( x, y, label=obj )
        if np.size( y ) > 0:
          agr_max_y = max( np.max( y ), agr_max_y )

        sorted_frames.sort( key=lambda line: line[2], reverse=True )
        if len( sorted_frames ) > 0:
          video_max_counts[obj].append( sorted_frames[0] )
        with open( os.path.join( video_subdir, obj + ranked_ext ), "w" ) as of:
          of.write( "# video_id, time_id, frame_id, detection_count\n" )
          for filename, frame_id, count in sorted_frames:
            frame_time = frame_id / frame_rate
            of.write( filename + "," + format_x(frame_time, show_ms=True) + ",")
            of.write( str(frame_id) + "," + str(count) + "\n" )

    agr_ax.set_ylim( ymin = 0 )
    agr_ax.set_xlim( xmin = 0 )
    if agr_max_y < 5:
      agr_ax.set_ylim( ymax = 5 )
    agr_ax.locator_params( axis='x', nbins = 7 )
    lgd = [ agr_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) ]
    agr_fig.set_size_inches( 10, 7 )
    agr_fig.savefig( os.path.join( output_directory, filename + ".png" ),
                     dpi=100, bbox_inches="tight", bbox_extra_artists=lgd )
    plt.close(agr_fig)

  # Write out aggregate information across all videos
  for obj in objects:
    with open( os.path.join( output_directory, "max_counts-" + obj + ".csv" ), "w" ) as of:
      of.write( "# video_id, time_id, frame_id, detection_count\n" )
      for filename, frame_id, count in video_max_counts[obj]:
        frame_time = frame_id / frame_rate
        of.write( filename + "," + format_x( frame_time, show_ms=True ) + "," )
        of.write( str( frame_id ) + "," + str( count ) + "\n" )


def print_and_exit( msg, code=1 ):
    print( msg )
    sys.exit( code )


def make_dir_if_not_exist( dirname ):
    if not os.path.exists( dirname ):
        os.makedirs( dirname, exist_ok=True )


# -------------------- DATA LOADING FUNCTIONS --------------------------

def load_pr_curve_csv( filepath ):
    """Load PR curve data from CSV file."""
    points = []
    ap = 0.0
    max_f1 = 0.0
    best_threshold = 0.0

    with open( filepath, 'r' ) as f:
        header = f.readline().strip().split( ',' )
        for line in f:
            parts = line.strip().split( ',' )
            if len( parts ) >= 4:
                point = {
                    'confidence': float( parts[0] ),
                    'recall': float( parts[1] ),
                    'precision': float( parts[2] ),
                    'f1': float( parts[3] ),
                }
                points.append( point )
                if point['f1'] > max_f1:
                    max_f1 = point['f1']
                    best_threshold = point['confidence']

    # Compute AP using 11-point interpolation
    if points:
        recalls = [p['recall'] for p in points]
        precisions = [p['precision'] for p in points]
        ap = compute_ap_11point( recalls, precisions )

    return {
        'points': points,
        'average_precision': ap,
        'max_f1': max_f1,
        'best_threshold': best_threshold
    }


def compute_ap_11point( recalls, precisions ):
    """Compute Average Precision using 11-point interpolation."""
    if not recalls or not precisions:
        return 0.0

    ap = 0.0
    for t in np.arange( 0.0, 1.1, 0.1 ):
        precisions_at_recall = [p for r, p in zip( recalls, precisions ) if r >= t]
        if precisions_at_recall:
            ap += max( precisions_at_recall )
    return ap / 11.0


def load_confusion_matrix_csv( filepath ):
    """Load confusion matrix from CSV file."""
    class_names = []
    matrix = []

    with open( filepath, 'r' ) as f:
        header = f.readline().strip().split( ',' )
        class_names = header[1:]  # Skip 'gt_class' column
        for line in f:
            parts = line.strip().split( ',' )
            if len( parts ) > 1:
                row_values = [int( x ) for x in parts[1:]]
                matrix.append( row_values )

    return class_names, np.array( matrix )


def load_roc_curve_csv(filepath):
    with open(filepath, newline='') as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(headers) < 3:
            raise ValueError('ROC CSV must contain confidence, x and recall columns')
        x_key = headers[1]
        points = [{'confidence': float(row[headers[0]]),
                   'false_alarms_per_frame' if x_key == 'false_alarms_per_frame' else 'fpr': float(row[headers[1]]),
                   'true_positive_rate': float(row[headers[2]])} for row in reader]
    return {'points': points}


def compute_auc( fprs, tprs ):
    """Compute Area Under Curve using trapezoidal rule."""
    if len( fprs ) < 2:
        return 0.0

    # Sort by FPR
    sorted_pairs = sorted( zip( fprs, tprs ) )
    fprs = [p[0] for p in sorted_pairs]
    tprs = [p[1] for p in sorted_pairs]

    auc = 0.0
    for i in range( 1, len( fprs ) ):
        auc += ( fprs[i] - fprs[i-1] ) * ( tprs[i] + tprs[i-1] ) / 2.0
    return auc


def load_metrics_csv( filepath ):
    """Load metrics from CSV file."""
    metrics = {}
    with open( filepath, 'r' ) as f:
        reader = csv.reader( f )
        for row in reader:
            if len( row ) >= 2 and not row[0].startswith( '#' ):
                try:
                    metrics[row[0]] = float( row[1] )
                except ValueError:
                    metrics[row[0]] = row[1]
    return metrics


def load_json_data( filepath ):
    """Load all plot data from JSON file."""
    with open( filepath, 'r' ) as f:
        return json.load( f )


# -------------------- PLOTTING FUNCTIONS ------------------------------

def plot_pr_curve( pr_data, output_path, title="Precision-Recall Curve" ):
    """Generate precision-recall curve plot."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots( figsize=DEFAULT_FIGSIZE )

    if isinstance( pr_data, dict ) and 'points' in pr_data:
        points = pr_data['points']
        if points:
            recalls = [p['recall'] for p in points]
            precisions = [p['precision'] for p in points]
            ap = pr_data.get( 'average_precision', 0 )
            max_f1 = pr_data.get( 'max_f1', 0 )
            ax.plot( recalls, precisions, 'b-', linewidth=2,
                     label=f'AP = {ap:.3f}, Max F1 = {max_f1:.3f}' )
            ax.legend( loc='lower left' )
    elif isinstance( pr_data, list ):
        recalls = [p['recall'] for p in pr_data]
        precisions = [p['precision'] for p in pr_data]
        ax.plot( recalls, precisions, 'b-', linewidth=2 )

    ax.set_xlabel( 'Recall', fontsize=12 )
    ax.set_ylabel( 'Precision', fontsize=12 )
    ax.set_title( title, fontsize=14 )
    ax.set_xlim( [0, 1] )
    ax.set_ylim( [0, 1] )
    ax.grid( True, alpha=0.3 )

    plt.tight_layout()
    plt.savefig( output_path, dpi=DEFAULT_DPI, bbox_inches='tight' )
    plt.close()
    print( f"Saved: {output_path}" )


def plot_multi_class_pr_curves( per_class_data, output_path,
                                 title="Per-Class Precision-Recall Curves" ):
    """Generate multi-class PR curve plot."""
    if not HAS_MATPLOTLIB or not per_class_data:
        return

    fig, ax = plt.subplots( figsize=(10, 8) )
    colors = plt.cm.tab10( np.linspace( 0, 1, min( 10, len( per_class_data ) ) ) )

    # Compute mean AP
    aps = []
    for idx, ( class_name, data ) in enumerate( per_class_data.items() ):
        if 'points' in data and data['points']:
            recalls = [p['recall'] for p in data['points']]
            precisions = [p['precision'] for p in data['points']]
            ap = data.get( 'average_precision', 0 )
            aps.append( ap )
            color = colors[idx % len( colors )]
            ax.plot( recalls, precisions, color=color, linewidth=2,
                     label=f'{class_name} (AP={ap:.3f})' )

    mean_ap = np.mean( aps ) if aps else 0
    ax.set_xlabel( 'Recall', fontsize=12 )
    ax.set_ylabel( 'Precision', fontsize=12 )
    ax.set_title( f'{title}\nmAP = {mean_ap:.3f}', fontsize=14 )
    ax.set_xlim( [0, 1] )
    ax.set_ylim( [0, 1] )
    ax.grid( True, alpha=0.3 )
    ax.legend( loc='lower left', fontsize=9 )

    plt.tight_layout()
    plt.savefig( output_path, dpi=DEFAULT_DPI, bbox_inches='tight' )
    plt.close()
    print( f"Saved: {output_path}" )


def plot_confusion_matrix( class_names, matrix, output_path,
                           title="Confusion Matrix", normalize=False ):
    """Generate confusion matrix heatmap."""
    if not HAS_MATPLOTLIB:
        return

    if normalize:
        row_sums = matrix.sum( axis=1, keepdims=True )
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix.astype( float ) / row_sums
        fmt = '.2f'
        cbar_label = 'Fraction'
    else:
        fmt = 'd'
        cbar_label = 'Count'

    fig_size = ( max( 8, len( class_names ) * 0.8 ),
                 max( 6, len( class_names ) * 0.6 ) )
    fig, ax = plt.subplots( figsize=fig_size )

    if HAS_SEABORN:
        sns.heatmap( matrix, annot=True, fmt=fmt, cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names,
                     ax=ax, cbar_kws={'label': cbar_label} )
    else:
        im = ax.imshow( matrix, cmap='Blues' )
        ax.set_xticks( np.arange( len( class_names ) ) )
        ax.set_yticks( np.arange( len( class_names ) ) )
        ax.set_xticklabels( class_names )
        ax.set_yticklabels( class_names )
        plt.setp( ax.get_xticklabels(), rotation=45, ha="right",
                  rotation_mode="anchor" )

        # Add text annotations
        thresh = matrix.max() / 2.0
        for i in range( len( class_names ) ):
            for j in range( len( class_names ) ):
                val = matrix[i, j]
                if normalize:
                    text = f'{val:.2f}'
                else:
                    text = str( int( val ) )
                ax.text( j, i, text, ha="center", va="center",
                         color="white" if val > thresh else "black" )

        plt.colorbar( im, ax=ax, label=cbar_label )

    ax.set_xlabel( 'Predicted Class', fontsize=12 )
    ax.set_ylabel( 'Ground Truth Class', fontsize=12 )
    ax.set_title( title, fontsize=14 )

    plt.tight_layout()
    plt.savefig( output_path, dpi=DEFAULT_DPI, bbox_inches='tight' )
    plt.close()
    print( f"Saved: {output_path}" )


def plot_roc_curve(roc_data, output_path, title="ROC Curve"):
    if not HAS_MATPLOTLIB:
        return
    points = roc_data.get('points', []) if isinstance(roc_data, dict) else roc_data
    per_frame = any('false_alarms_per_frame' in p for p in points)
    x = [p.get('false_alarms_per_frame', p.get('fpr', 0)) for p in points]
    y = [p.get('true_positive_rate', p.get('tpr', 0)) for p in points]
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.plot(x, y, 'b-', linewidth=2)
    if not per_frame:
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False alarms per frame' if per_frame else 'False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title(title, fontsize=14)
    finite_x = [v for v in x if v is not None and np.isfinite(v)]
    ax.set_xlim(0, max(1, max(finite_x, default=0) * 1.05) if per_frame else 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_histogram( data, output_path, title, xlabel, ylabel='Count',
                    bins=None, color='steelblue' ):
    """Generate histogram plot."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots( figsize=DEFAULT_FIGSIZE )

    if isinstance( data, list ):
        if bins:
            ax.bar( range( len( data ) ), data, width=0.8,
                    color=color, edgecolor='black' )
            ax.set_xticks( range( len( bins ) ) )
            ax.set_xticklabels( bins, rotation=45, ha='right' )
        else:
            ax.bar( range( len( data ) ), data, width=0.8,
                    color=color, edgecolor='black' )
    elif isinstance( data, dict ):
        keys = sorted( data.keys(), key=lambda x: int( x ) if str( x ).isdigit() else x )
        values = [data[k] for k in keys]
        ax.bar( range( len( keys ) ), values, width=0.8,
                color=color, edgecolor='black' )
        ax.set_xticks( range( len( keys ) ) )
        ax.set_xticklabels( keys, rotation=45 if len( keys ) > 10 else 0 )

    ax.set_xlabel( xlabel, fontsize=12 )
    ax.set_ylabel( ylabel, fontsize=12 )
    ax.set_title( title, fontsize=14 )
    ax.yaxis.set_major_locator( MaxNLocator( integer=True ) )

    plt.tight_layout()
    plt.savefig( output_path, dpi=DEFAULT_DPI, bbox_inches='tight' )
    plt.close()
    print( f"Saved: {output_path}" )


def plot_iou_histogram( iou_data, output_path ):
    """Generate IoU distribution histogram."""
    bins = [f'{i*5}-{(i+1)*5}%' for i in range( 20 )]
    plot_histogram( iou_data, output_path,
                    title='IoU Distribution of True Positives',
                    xlabel='IoU Range', bins=bins )


def plot_track_length_histogram( track_data, output_path ):
    """Generate track length histogram."""
    plot_histogram( track_data, output_path,
                    title='Track Length Distribution',
                    xlabel='Track Length (frames)' )


def plot_track_purity_histogram( purity_data, output_path ):
    """Generate track purity histogram."""
    bins = [f'{i*10}-{(i+1)*10}%' for i in range( 10 )]
    plot_histogram( purity_data, output_path,
                    title='Track Purity Distribution',
                    xlabel='Purity Range', bins=bins,
                    color='forestgreen' )


def plot_track_continuity_histogram( continuity_data, output_path ):
    """Generate track continuity histogram."""
    bins = [f'{i*10}-{(i+1)*10}%' for i in range( 10 )]
    plot_histogram( continuity_data, output_path,
                    title='Track Continuity Distribution',
                    xlabel='Continuity Range', bins=bins,
                    color='darkorange' )


def plot_detection_metrics_summary( metrics, output_path ):
    """Generate detection metrics summary bar chart."""
    if not HAS_MATPLOTLIB:
        return

    metric_names = ['precision', 'recall', 'f1_score', 'average_precision',
                    'ap50', 'ap75', 'ap50_95']
    display_names = ['Precision', 'Recall', 'F1', 'AP', 'AP@50', 'AP@75', 'AP@50:95']

    values = []
    labels = []
    for name, display in zip( metric_names, display_names ):
        if name in metrics:
            values.append( metrics[name] )
            labels.append( display )

    if not values:
        return

    fig, ax = plt.subplots( figsize=(10, 6) )
    bars = ax.bar( range( len( values ) ), values, color='steelblue', edgecolor='black' )

    # Add value labels on bars
    for bar, val in zip( bars, values ):
        height = bar.get_height()
        ax.text( bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10 )

    ax.set_xticks( range( len( labels ) ) )
    ax.set_xticklabels( labels, fontsize=11 )
    ax.set_ylabel( 'Score', fontsize=12 )
    ax.set_title( 'Detection Metrics Summary', fontsize=14 )
    ax.set_ylim( [0, 1.1] )
    ax.grid( True, alpha=0.3, axis='y' )

    plt.tight_layout()
    plt.savefig( output_path, dpi=DEFAULT_DPI, bbox_inches='tight' )
    plt.close()
    print( f"Saved: {output_path}" )


def plot_mot_metrics_summary( metrics, output_path ):
    """Generate MOT metrics summary bar chart."""
    if not HAS_MATPLOTLIB:
        return

    metric_names = ['mota', 'motp', 'idf1', 'idp', 'idr']
    display_names = ['MOTA', 'MOTP', 'IDF1', 'IDP', 'IDR']

    values = []
    labels = []
    for name, display in zip( metric_names, display_names ):
        if name in metrics:
            values.append( metrics[name] )
            labels.append( display )

    if not values:
        return

    fig, ax = plt.subplots( figsize=(10, 6) )

    # Color bars based on sign (MOTA can be negative)
    colors = ['forestgreen' if v >= 0 else 'crimson' for v in values]
    bars = ax.bar( range( len( values ) ), values, color=colors, edgecolor='black' )

    for bar, val in zip( bars, values ):
        height = bar.get_height()
        va = 'bottom' if val >= 0 else 'top'
        ax.text( bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.3f}', ha='center', va=va, fontsize=10 )

    ax.set_xticks( range( len( labels ) ) )
    ax.set_xticklabels( labels, fontsize=11 )
    ax.set_ylabel( 'Score', fontsize=12 )
    ax.set_title( 'MOT Metrics Summary', fontsize=14 )
    ax.axhline( y=0, color='black', linestyle='-', linewidth=0.5 )
    ax.grid( True, alpha=0.3, axis='y' )

    plt.tight_layout()
    plt.savefig( output_path, dpi=DEFAULT_DPI, bbox_inches='tight' )
    plt.close()
    print( f"Saved: {output_path}" )


def plot_hota_metrics_summary( metrics, output_path ):
    """Generate HOTA metrics summary bar chart."""
    if not HAS_MATPLOTLIB:
        return

    metric_names = ['hota', 'deta', 'assa', 'loca']
    display_names = ['HOTA', 'DetA', 'AssA', 'LocA']

    values = []
    labels = []
    for name, display in zip( metric_names, display_names ):
        if name in metrics:
            values.append( metrics[name] )
            labels.append( display )

    if not values:
        return

    fig, ax = plt.subplots( figsize=(8, 6) )
    bars = ax.bar( range( len( values ) ), values, color='darkorange', edgecolor='black' )

    for bar, val in zip( bars, values ):
        height = bar.get_height()
        ax.text( bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10 )

    ax.set_xticks( range( len( labels ) ) )
    ax.set_xticklabels( labels, fontsize=11 )
    ax.set_ylabel( 'Score', fontsize=12 )
    ax.set_title( 'HOTA Metrics Summary', fontsize=14 )
    ax.set_ylim( [0, 1.1] )
    ax.grid( True, alpha=0.3, axis='y' )

    plt.tight_layout()
    plt.savefig( output_path, dpi=DEFAULT_DPI, bbox_inches='tight' )
    plt.close()
    print( f"Saved: {output_path}" )


def plot_id_switch_breakdown( metrics, output_path ):
    """Generate ID switch breakdown bar chart (transfer, ascend, migrate)."""
    if not HAS_MATPLOTLIB:
        return

    metric_names = ['id_switches', 'num_transfer', 'num_ascend', 'num_migrate',
                    'fragmentations']
    display_names = ['ID Switches', 'Transfers', 'Ascends', 'Migrates',
                     'Fragmentations']

    values = []
    labels = []
    for name, display in zip( metric_names, display_names ):
        if name in metrics and metrics[name] > 0:
            values.append( metrics[name] )
            labels.append( display )

    if not values:
        return

    fig, ax = plt.subplots( figsize=(10, 6) )
    colors = ['crimson', 'coral', 'salmon', 'lightsalmon', 'orange']
    bars = ax.bar( range( len( values ) ), values,
                   color=colors[:len( values )], edgecolor='black' )

    for bar, val in zip( bars, values ):
        height = bar.get_height()
        ax.text( bar.get_x() + bar.get_width() / 2., height,
                 f'{int(val)}', ha='center', va='bottom', fontsize=10 )

    ax.set_xticks( range( len( labels ) ) )
    ax.set_xticklabels( labels, fontsize=11 )
    ax.set_ylabel( 'Count', fontsize=12 )
    ax.set_title( 'ID Switch Breakdown', fontsize=14 )
    ax.yaxis.set_major_locator( MaxNLocator( integer=True ) )
    ax.grid( True, alpha=0.3, axis='y' )

    plt.tight_layout()
    plt.savefig( output_path, dpi=DEFAULT_DPI, bbox_inches='tight' )
    plt.close()
    print( f"Saved: {output_path}" )


def plot_track_quality_summary( metrics, output_path ):
    """Generate track quality summary (MT, PT, ML)."""
    if not HAS_MATPLOTLIB:
        return

    metric_names = ['mostly_tracked', 'partially_tracked', 'mostly_lost']
    display_names = ['Mostly Tracked', 'Partially Tracked', 'Mostly Lost']
    colors = ['forestgreen', 'gold', 'crimson']

    values = []
    labels = []
    plot_colors = []
    for name, display, color in zip( metric_names, display_names, colors ):
        if name in metrics:
            values.append( metrics[name] )
            labels.append( display )
            plot_colors.append( color )

    if not values:
        return

    fig, ax = plt.subplots( figsize=(8, 6) )

    # Create pie chart if we have all three, otherwise bar chart
    if len( values ) == 3 and sum( values ) > 0:
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=plot_colors,
            autopct='%1.1f%%', startangle=90 )
        ax.set_title( 'Track Quality Distribution', fontsize=14 )
    else:
        bars = ax.bar( range( len( values ) ), values,
                       color=plot_colors, edgecolor='black' )
        for bar, val in zip( bars, values ):
            height = bar.get_height()
            ax.text( bar.get_x() + bar.get_width() / 2., height,
                     f'{int(val)}', ha='center', va='bottom', fontsize=10 )
        ax.set_xticks( range( len( labels ) ) )
        ax.set_xticklabels( labels, fontsize=11 )
        ax.set_ylabel( 'Count', fontsize=12 )
        ax.set_title( 'Track Quality Summary', fontsize=14 )
        ax.yaxis.set_major_locator( MaxNLocator( integer=True ) )

    plt.tight_layout()
    plt.savefig( output_path, dpi=DEFAULT_DPI, bbox_inches='tight' )
    plt.close()
    print( f"Saved: {output_path}" )


# -------------------- MAIN GENERATION FUNCTIONS -----------------------

def generate_all_plots_from_json( json_path, output_dir ):
    """Generate all plots from JSON data file."""
    data = load_json_data( json_path )
    # Accept both the evaluator export and the score applet's metrics JSON.
    if 'pr_curve' in data:
        data['overall_pr_curve'] = data['pr_curve']
    if 'roc_curve' in data:
        data['overall_roc_curve'] = data['roc_curve']
    if 'metrics' not in data:
        data['metrics'] = {k: v for k, v in data.items() if isinstance(v, (int, float))}
    make_dir_if_not_exist( output_dir )

    # Overall PR curve
    if 'overall_pr_curve' in data:
        plot_pr_curve( data['overall_pr_curve'],
                       os.path.join( output_dir, 'pr_curve_overall.png' ) )

    # Per-class PR curves
    if 'per_class_pr_curves' in data and data['per_class_pr_curves']:
        plot_multi_class_pr_curves( data['per_class_pr_curves'],
                                     os.path.join( output_dir, 'pr_curves_per_class.png' ) )

    # Confusion matrix
    if 'confusion_matrix' in data:
        cm = data['confusion_matrix']
        if 'class_names' in cm and 'matrix' in cm:
            matrix = np.array( cm['matrix'] )
            plot_confusion_matrix( cm['class_names'], matrix,
                                   os.path.join( output_dir, 'confusion_matrix.png' ) )
            plot_confusion_matrix( cm['class_names'], matrix,
                                   os.path.join( output_dir, 'confusion_matrix_normalized.png' ),
                                   title='Normalized Confusion Matrix', normalize=True )

    # ROC curve
    if 'overall_roc_curve' in data:
        plot_roc_curve( data['overall_roc_curve'],
                        os.path.join( output_dir, 'roc_curve.png' ) )

    # IoU histogram
    if 'iou_histogram' in data:
        plot_iou_histogram( data['iou_histogram'],
                            os.path.join( output_dir, 'iou_histogram.png' ) )

    # Track length histogram
    if 'track_length_histogram' in data:
        plot_track_length_histogram( data['track_length_histogram'],
                                      os.path.join( output_dir, 'track_length_histogram.png' ) )

    # Track purity histogram
    if 'track_purity_histogram' in data:
        plot_track_purity_histogram( data['track_purity_histogram'],
                                      os.path.join( output_dir, 'track_purity_histogram.png' ) )

    # Track continuity histogram
    if 'track_continuity_histogram' in data:
        plot_track_continuity_histogram( data['track_continuity_histogram'],
                                          os.path.join( output_dir, 'track_continuity_histogram.png' ) )

    # Metrics summaries
    if 'metrics' in data:
        metrics = data['metrics']
        plot_detection_metrics_summary( metrics,
                                         os.path.join( output_dir, 'detection_metrics.png' ) )
        plot_mot_metrics_summary( metrics,
                                   os.path.join( output_dir, 'mot_metrics.png' ) )
        plot_hota_metrics_summary( metrics,
                                    os.path.join( output_dir, 'hota_metrics.png' ) )
        plot_id_switch_breakdown( metrics,
                                   os.path.join( output_dir, 'id_switch_breakdown.png' ) )
        plot_track_quality_summary( metrics,
                                     os.path.join( output_dir, 'track_quality.png' ) )


def generate_all_plots_from_csv_dir( input_dir, output_dir ):
    """Generate all plots from CSV files in directory."""
    make_dir_if_not_exist( output_dir )
    input_path = Path( input_dir )

    # Overall PR curve
    pr_overall = input_path / 'pr_curve_overall.csv'
    if pr_overall.exists():
        data = load_pr_curve_csv( pr_overall )
        plot_pr_curve( data, os.path.join( output_dir, 'pr_curve_overall.png' ) )

    # Per-class PR curves
    per_class_data = {}
    for pr_file in input_path.glob( 'pr_curve_*.csv' ):
        if pr_file.name != 'pr_curve_overall.csv':
            class_name = pr_file.stem.replace( 'pr_curve_', '' )
            per_class_data[class_name] = load_pr_curve_csv( pr_file )

    if per_class_data:
        plot_multi_class_pr_curves( per_class_data,
                                     os.path.join( output_dir, 'pr_curves_per_class.png' ) )

    # Confusion matrix
    cm_file = input_path / 'confusion_matrix.csv'
    if cm_file.exists():
        class_names, matrix = load_confusion_matrix_csv( cm_file )
        plot_confusion_matrix( class_names, matrix,
                               os.path.join( output_dir, 'confusion_matrix.png' ) )
        plot_confusion_matrix( class_names, matrix,
                               os.path.join( output_dir, 'confusion_matrix_normalized.png' ),
                               title='Normalized Confusion Matrix', normalize=True )

    # ROC curve
    roc_file = input_path / 'roc_curve_overall.csv'
    if roc_file.exists():
        data = load_roc_curve_csv( roc_file )
        plot_roc_curve( data, os.path.join( output_dir, 'roc_curve.png' ) )

    # Metrics file
    metrics_file = input_path / 'metrics.csv'
    if metrics_file.exists():
        metrics = load_metrics_csv( metrics_file )
        plot_detection_metrics_summary( metrics,
                                         os.path.join( output_dir, 'detection_metrics.png' ) )
        plot_mot_metrics_summary( metrics,
                                   os.path.join( output_dir, 'mot_metrics.png' ) )
        plot_hota_metrics_summary( metrics,
                                    os.path.join( output_dir, 'hota_metrics.png' ) )
        plot_id_switch_breakdown( metrics,
                                   os.path.join( output_dir, 'id_switch_breakdown.png' ) )
        plot_track_quality_summary( metrics,
                                     os.path.join( output_dir, 'track_quality.png' ) )


# -------------------------- MAIN FUNCTION -----------------------------

def main( argv=None ):
    global DEFAULT_DPI

    parser = argparse.ArgumentParser(
        prog='viame plot',
        description='Plot detection counts per frame, or VIAME evaluation results.' )
    sub = parser.add_subparsers( dest='command', metavar='command' )
    sub.required = True

    p = sub.add_parser( 'detections',
        help='Plot detection counts per frame for a folder of VIAME CSV files' )
    p.add_argument( 'objects',
        help='Comma separated list of categories to plot' )
    p.add_argument( 'threshold', type=float,
        help='Detection confidence threshold' )
    p.add_argument( 'frame_rate', type=float,
        help='Frame rate of the source video, used to label the time axis' )
    p.add_argument( 'smooth', type=int, nargs='?', default=1,
        help='Smoothing window in frames (default: 1)' )
    p.add_argument( '-input', '-i', dest='input', default='.',
        help='Folder holding the VIAME CSV files (default: current folder)' )
    p.add_argument( '-output', '-o', dest='output', default='plots',
        help='Output folder for the plots (default: plots)' )
    p.add_argument( '-ext', dest='ext', default='.csv',
        help='Extension of the detection files (default: .csv)' )
    p.add_argument( '--top-category-only', dest='top_category_only', action='store_true',
        help='Count only each detection\'s top category' )

    p = sub.add_parser( 'eval', help='Generate plots from VIAME evaluation results' )
    p.add_argument( '-input', '-i', dest='input', required=True,
        help='Input directory with CSV files or JSON file' )
    p.add_argument( '-output', '-o', dest='output', default='./plots',
        help='Output directory for plots (default: ./plots)' )
    p.add_argument( '-dpi', type=int, default=DEFAULT_DPI,
        help=f'Output image DPI (default: {DEFAULT_DPI})' )

    args = parser.parse_args( argv )

    if not HAS_MATPLOTLIB:
        print_and_exit( "Error: matplotlib is required. Install with: pip install matplotlib" )

    if args.command == 'detections':
        detection_plot( args.input, args.output, args.objects.split( "," ),
                        args.threshold, args.frame_rate, args.smooth,
                        ext=args.ext, top_category_only=args.top_category_only )
        print( f"\nPlots saved to: {args.output}" )
        return 0

    DEFAULT_DPI = args.dpi
    input_path = Path( args.input )

    if input_path.is_file() and input_path.suffix == '.json':
        print( f"Loading JSON data from: {args.input}" )
        generate_all_plots_from_json( args.input, args.output )
    elif input_path.is_dir():
        print( f"Loading CSV files from: {args.input}" )
        generate_all_plots_from_csv_dir( args.input, args.output )
    else:
        print_and_exit( f"Error: {args.input} is not a valid JSON file or directory" )

    print( f"\nAll plots saved to: {args.output}" )
    return 0


if __name__ == "__main__":
    sys.exit( main() )
