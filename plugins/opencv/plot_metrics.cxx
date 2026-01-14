/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Implementation of plotting utilities for evaluation metrics

#include "plot_metrics.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace viame {

// =============================================================================
// Helper functions
// =============================================================================

namespace {

std::string format_double( double value, int precision = 2 )
{
  std::ostringstream ss;
  ss << std::fixed << std::setprecision( precision ) << value;
  return ss.str();
}

// Interpolate between two colors based on t in [0, 1]
cv::Scalar interpolate_color( const cv::Scalar& low, const cv::Scalar& high, double t )
{
  t = std::max( 0.0, std::min( 1.0, t ) );
  return cv::Scalar(
    low[0] + t * ( high[0] - low[0] ),
    low[1] + t * ( high[1] - low[1] ),
    low[2] + t * ( high[2] - low[2] )
  );
}

// Get text size with baseline
cv::Size get_text_size( const std::string& text, int font_face,
                        double font_scale, int thickness, int* baseline = nullptr )
{
  int base = 0;
  cv::Size size = cv::getTextSize( text, font_face, font_scale, thickness, &base );
  if( baseline )
  {
    *baseline = base;
  }
  return size;
}

} // anonymous namespace

// =============================================================================
// plot_color_scheme implementation
// =============================================================================

cv::Scalar
plot_color_scheme::get_line_color( size_t index ) const
{
  if( line_colors.empty() )
  {
    return cv::Scalar( 0, 0, 255 ); // Default red
  }
  return line_colors[ index % line_colors.size() ];
}

// =============================================================================
// metrics_plotter::priv implementation
// =============================================================================

class metrics_plotter::priv
{
public:
  plot_config config;

  // Font settings
  static constexpr int font_face = cv::FONT_HERSHEY_SIMPLEX;

  // Computed plot area bounds
  int plot_left() const { return config.margin_left; }
  int plot_right() const { return config.width - config.margin_right; }
  int plot_top() const { return config.margin_top; }
  int plot_bottom() const { return config.height - config.margin_bottom; }
  int plot_width() const { return plot_right() - plot_left(); }
  int plot_height() const { return plot_bottom() - plot_top(); }

  // Coordinate transformations
  cv::Point data_to_pixel( double x, double y ) const
  {
    double x_range = config.x_max - config.x_min;
    double y_range = config.y_max - config.y_min;

    int px = plot_left() + static_cast< int >(
      ( x - config.x_min ) / x_range * plot_width() );
    int py = plot_bottom() - static_cast< int >(
      ( y - config.y_min ) / y_range * plot_height() );

    return cv::Point( px, py );
  }

  // Create base canvas with background
  cv::Mat create_canvas() const
  {
    cv::Mat canvas( config.height, config.width, CV_8UC3, config.colors.background );

    // Draw plot area background
    cv::rectangle( canvas,
      cv::Point( plot_left(), plot_top() ),
      cv::Point( plot_right(), plot_bottom() ),
      config.colors.plot_area, cv::FILLED );

    return canvas;
  }

  // Draw grid lines
  void draw_grid( cv::Mat& canvas ) const
  {
    if( !config.show_grid )
    {
      return;
    }

    // Vertical grid lines
    for( int i = 0; i <= config.num_x_ticks - 1; ++i )
    {
      double t = static_cast< double >( i ) / ( config.num_x_ticks - 1 );
      double x = config.x_min + t * ( config.x_max - config.x_min );
      cv::Point pt = data_to_pixel( x, config.y_min );
      cv::line( canvas,
        cv::Point( pt.x, plot_top() ),
        cv::Point( pt.x, plot_bottom() ),
        config.colors.grid_color, 1, cv::LINE_AA );
    }

    // Horizontal grid lines
    for( int i = 0; i <= config.num_y_ticks - 1; ++i )
    {
      double t = static_cast< double >( i ) / ( config.num_y_ticks - 1 );
      double y = config.y_min + t * ( config.y_max - config.y_min );
      cv::Point pt = data_to_pixel( config.x_min, y );
      cv::line( canvas,
        cv::Point( plot_left(), pt.y ),
        cv::Point( plot_right(), pt.y ),
        config.colors.grid_color, 1, cv::LINE_AA );
    }
  }

  // Draw axes
  void draw_axes( cv::Mat& canvas ) const
  {
    // Draw axis lines
    cv::rectangle( canvas,
      cv::Point( plot_left(), plot_top() ),
      cv::Point( plot_right(), plot_bottom() ),
      config.colors.axis_color, 1, cv::LINE_AA );

    // Draw tick marks and labels on X axis
    for( int i = 0; i < config.num_x_ticks; ++i )
    {
      double t = static_cast< double >( i ) / ( config.num_x_ticks - 1 );
      double x = config.x_min + t * ( config.x_max - config.x_min );
      cv::Point pt = data_to_pixel( x, config.y_min );

      // Tick mark
      cv::line( canvas,
        cv::Point( pt.x, plot_bottom() ),
        cv::Point( pt.x, plot_bottom() + 5 ),
        config.colors.axis_color, 1, cv::LINE_AA );

      // Label
      std::string label = format_double( x, 1 );
      int baseline = 0;
      cv::Size text_size = get_text_size( label, font_face,
        config.font_scale, config.font_thickness, &baseline );
      cv::putText( canvas, label,
        cv::Point( pt.x - text_size.width / 2, plot_bottom() + 20 ),
        font_face, config.font_scale, config.colors.text_color,
        config.font_thickness, cv::LINE_AA );
    }

    // Draw tick marks and labels on Y axis
    for( int i = 0; i < config.num_y_ticks; ++i )
    {
      double t = static_cast< double >( i ) / ( config.num_y_ticks - 1 );
      double y = config.y_min + t * ( config.y_max - config.y_min );
      cv::Point pt = data_to_pixel( config.x_min, y );

      // Tick mark
      cv::line( canvas,
        cv::Point( plot_left() - 5, pt.y ),
        cv::Point( plot_left(), pt.y ),
        config.colors.axis_color, 1, cv::LINE_AA );

      // Label
      std::string label = format_double( y, 1 );
      int baseline = 0;
      cv::Size text_size = get_text_size( label, font_face,
        config.font_scale, config.font_thickness, &baseline );
      cv::putText( canvas, label,
        cv::Point( plot_left() - text_size.width - 10, pt.y + text_size.height / 2 ),
        font_face, config.font_scale, config.colors.text_color,
        config.font_thickness, cv::LINE_AA );
    }
  }

  // Draw title and axis labels
  void draw_labels( cv::Mat& canvas ) const
  {
    // Title
    if( !config.title.empty() )
    {
      int baseline = 0;
      cv::Size text_size = get_text_size( config.title, font_face,
        config.title_font_scale, config.font_thickness + 1, &baseline );
      cv::putText( canvas, config.title,
        cv::Point( ( config.width - text_size.width ) / 2, 30 ),
        font_face, config.title_font_scale, config.colors.title_color,
        config.font_thickness + 1, cv::LINE_AA );
    }

    // X axis label
    if( !config.x_label.empty() )
    {
      int baseline = 0;
      cv::Size text_size = get_text_size( config.x_label, font_face,
        config.font_scale, config.font_thickness, &baseline );
      cv::putText( canvas, config.x_label,
        cv::Point( ( config.width - text_size.width ) / 2,
                   config.height - 10 ),
        font_face, config.font_scale, config.colors.text_color,
        config.font_thickness, cv::LINE_AA );
    }

    // Y axis label (rotated - draw vertically)
    if( !config.y_label.empty() )
    {
      int baseline = 0;
      cv::Size text_size = get_text_size( config.y_label, font_face,
        config.font_scale, config.font_thickness, &baseline );

      // Create temporary image for rotation
      cv::Mat text_img( text_size.width + 10, text_size.height + 10, CV_8UC3,
                        config.colors.background );
      cv::putText( text_img, config.y_label,
        cv::Point( 5, text_size.height + 2 ),
        font_face, config.font_scale, config.colors.text_color,
        config.font_thickness, cv::LINE_AA );

      // Rotate 90 degrees counter-clockwise
      cv::Mat rotated;
      cv::rotate( text_img, rotated, cv::ROTATE_90_COUNTERCLOCKWISE );

      // Copy to main canvas
      int x_pos = 10;
      int y_pos = ( config.height - rotated.rows ) / 2;
      if( y_pos >= 0 && y_pos + rotated.rows <= canvas.rows &&
          x_pos >= 0 && x_pos + rotated.cols <= canvas.cols )
      {
        rotated.copyTo( canvas( cv::Rect( x_pos, y_pos, rotated.cols, rotated.rows ) ) );
      }
    }
  }

  // Draw a line curve
  void draw_curve( cv::Mat& canvas,
                   const std::vector< cv::Point >& points,
                   const cv::Scalar& color ) const
  {
    if( points.size() < 2 )
    {
      return;
    }

    for( size_t i = 1; i < points.size(); ++i )
    {
      cv::line( canvas, points[i-1], points[i], color,
                config.line_thickness, cv::LINE_AA );
    }

    if( config.show_points )
    {
      for( const auto& pt : points )
      {
        cv::circle( canvas, pt, config.point_radius, color, cv::FILLED, cv::LINE_AA );
      }
    }
  }

  // Draw legend
  void draw_legend( cv::Mat& canvas,
                    const std::vector< std::pair< std::string, cv::Scalar > >& entries ) const
  {
    if( !config.show_legend || entries.empty() )
    {
      return;
    }

    // Calculate legend size
    int max_text_width = 0;
    int line_height = 0;
    int baseline = 0;

    for( const auto& entry : entries )
    {
      cv::Size text_size = get_text_size( entry.first, font_face,
        config.font_scale, config.font_thickness, &baseline );
      max_text_width = std::max( max_text_width, text_size.width );
      line_height = std::max( line_height, text_size.height + baseline );
    }

    int legend_width = max_text_width + 40; // Line sample + padding
    int legend_height = static_cast< int >( entries.size() ) * ( line_height + 5 ) + 10;

    // Determine position
    int legend_x, legend_y;
    switch( config.legend_position )
    {
      case 1: // Top-left
        legend_x = plot_left() + 10;
        legend_y = plot_top() + 10;
        break;
      case 2: // Top-right
        legend_x = plot_right() - legend_width - 10;
        legend_y = plot_top() + 10;
        break;
      case 3: // Bottom-left
        legend_x = plot_left() + 10;
        legend_y = plot_bottom() - legend_height - 10;
        break;
      case 4: // Bottom-right
        legend_x = plot_right() - legend_width - 10;
        legend_y = plot_bottom() - legend_height - 10;
        break;
      default: // Auto - top-right for most plots
        legend_x = plot_right() - legend_width - 10;
        legend_y = plot_top() + 10;
        break;
    }

    // Draw legend background
    cv::rectangle( canvas,
      cv::Point( legend_x, legend_y ),
      cv::Point( legend_x + legend_width, legend_y + legend_height ),
      cv::Scalar( 255, 255, 255 ), cv::FILLED );
    cv::rectangle( canvas,
      cv::Point( legend_x, legend_y ),
      cv::Point( legend_x + legend_width, legend_y + legend_height ),
      config.colors.axis_color, 1, cv::LINE_AA );

    // Draw entries
    int y_offset = legend_y + line_height + 2;
    for( const auto& entry : entries )
    {
      // Color line sample
      cv::line( canvas,
        cv::Point( legend_x + 5, y_offset - line_height / 2 + 2 ),
        cv::Point( legend_x + 25, y_offset - line_height / 2 + 2 ),
        entry.second, config.line_thickness, cv::LINE_AA );

      // Text
      cv::putText( canvas, entry.first,
        cv::Point( legend_x + 30, y_offset ),
        font_face, config.font_scale, config.colors.text_color,
        config.font_thickness, cv::LINE_AA );

      y_offset += line_height + 5;
    }
  }

  // Draw annotation text on plot
  void draw_annotation( cv::Mat& canvas, const std::string& text,
                        int x, int y, const cv::Scalar& color ) const
  {
    // Draw with slight shadow for readability
    cv::putText( canvas, text, cv::Point( x + 1, y + 1 ),
      font_face, config.font_scale, cv::Scalar( 200, 200, 200 ),
      config.font_thickness, cv::LINE_AA );
    cv::putText( canvas, text, cv::Point( x, y ),
      font_face, config.font_scale, color,
      config.font_thickness, cv::LINE_AA );
  }
};

// =============================================================================
// metrics_plotter implementation
// =============================================================================

metrics_plotter::metrics_plotter()
  : d( new priv )
{
}

metrics_plotter::~metrics_plotter() = default;

metrics_plotter::metrics_plotter( metrics_plotter&& ) noexcept = default;

metrics_plotter& metrics_plotter::operator=( metrics_plotter&& ) noexcept = default;

void
metrics_plotter::set_config( const plot_config& config )
{
  d->config = config;
}

plot_config
metrics_plotter::get_config() const
{
  return d->config;
}

// -----------------------------------------------------------------------------
// PR Curve Rendering
// -----------------------------------------------------------------------------

cv::Mat
metrics_plotter::render_pr_curve( const pr_curve_data& curve )
{
  // Set up config for PR curve
  plot_config cfg = d->config;
  if( cfg.title.empty() )
  {
    cfg.title = curve.class_name.empty() ? "Precision-Recall Curve" :
      "Precision-Recall Curve: " + curve.class_name;
  }
  if( cfg.x_label.empty() ) cfg.x_label = "Recall";
  if( cfg.y_label.empty() ) cfg.y_label = "Precision";

  d->config = cfg;

  cv::Mat canvas = d->create_canvas();
  d->draw_grid( canvas );
  d->draw_axes( canvas );
  d->draw_labels( canvas );

  // Convert data to pixel coordinates
  std::vector< cv::Point > points;
  for( const auto& pt : curve.points )
  {
    points.push_back( d->data_to_pixel( pt.recall, pt.precision ) );
  }

  // Draw curve
  cv::Scalar line_color = d->config.colors.get_line_color( 0 );
  d->draw_curve( canvas, points, line_color );

  // Draw annotations
  if( d->config.show_auc && curve.average_precision > 0 )
  {
    std::string ap_text = "AP = " + format_double( curve.average_precision, 3 );
    d->draw_annotation( canvas, ap_text, d->plot_left() + 10, d->plot_top() + 25, line_color );
  }

  if( d->config.show_best_threshold && curve.max_f1 > 0 )
  {
    std::string f1_text = "Max F1 = " + format_double( curve.max_f1, 3 ) +
      " @ conf=" + format_double( curve.best_threshold, 2 );
    d->draw_annotation( canvas, f1_text, d->plot_left() + 10, d->plot_top() + 45, line_color );
  }

  return canvas;
}

cv::Mat
metrics_plotter::render_pr_curves( const std::map< std::string, pr_curve_data >& curves )
{
  // Set up config
  plot_config cfg = d->config;
  if( cfg.title.empty() ) cfg.title = "Precision-Recall Curves";
  if( cfg.x_label.empty() ) cfg.x_label = "Recall";
  if( cfg.y_label.empty() ) cfg.y_label = "Precision";

  d->config = cfg;

  cv::Mat canvas = d->create_canvas();
  d->draw_grid( canvas );
  d->draw_axes( canvas );
  d->draw_labels( canvas );

  // Draw each curve
  std::vector< std::pair< std::string, cv::Scalar > > legend_entries;
  size_t color_idx = 0;

  for( const auto& kv : curves )
  {
    const auto& name = kv.first;
    const auto& curve = kv.second;

    std::vector< cv::Point > points;
    for( const auto& pt : curve.points )
    {
      points.push_back( d->data_to_pixel( pt.recall, pt.precision ) );
    }

    cv::Scalar color = d->config.colors.get_line_color( color_idx++ );
    d->draw_curve( canvas, points, color );

    std::string label = name;
    if( d->config.show_auc && curve.average_precision > 0 )
    {
      label += " (AP=" + format_double( curve.average_precision, 2 ) + ")";
    }
    legend_entries.push_back( { label, color } );
  }

  d->draw_legend( canvas, legend_entries );

  return canvas;
}

// -----------------------------------------------------------------------------
// ROC Curve Rendering
// -----------------------------------------------------------------------------

cv::Mat
metrics_plotter::render_roc_curve( const roc_curve_data& curve )
{
  // Set up config for ROC curve
  plot_config cfg = d->config;
  if( cfg.title.empty() )
  {
    cfg.title = curve.class_name.empty() ? "ROC Curve" :
      "ROC Curve: " + curve.class_name;
  }
  if( cfg.x_label.empty() ) cfg.x_label = "False Positive Rate";
  if( cfg.y_label.empty() ) cfg.y_label = "True Positive Rate";

  d->config = cfg;

  cv::Mat canvas = d->create_canvas();
  d->draw_grid( canvas );

  // Draw diagonal reference line (random classifier)
  cv::Point diag_start = d->data_to_pixel( 0.0, 0.0 );
  cv::Point diag_end = d->data_to_pixel( 1.0, 1.0 );
  cv::line( canvas, diag_start, diag_end,
            cv::Scalar( 180, 180, 180 ), 1, cv::LINE_AA );

  d->draw_axes( canvas );
  d->draw_labels( canvas );

  // Convert data to pixel coordinates
  std::vector< cv::Point > points;
  for( const auto& pt : curve.points )
  {
    points.push_back( d->data_to_pixel( pt.false_positive_rate, pt.true_positive_rate ) );
  }

  // Draw curve
  cv::Scalar line_color = d->config.colors.get_line_color( 0 );
  d->draw_curve( canvas, points, line_color );

  // Draw AUC annotation
  if( d->config.show_auc && curve.auc > 0 )
  {
    std::string auc_text = "AUC = " + format_double( curve.auc, 3 );
    d->draw_annotation( canvas, auc_text, d->plot_left() + 10, d->plot_top() + 25, line_color );
  }

  return canvas;
}

cv::Mat
metrics_plotter::render_roc_curves( const std::map< std::string, roc_curve_data >& curves )
{
  // Set up config
  plot_config cfg = d->config;
  if( cfg.title.empty() ) cfg.title = "ROC Curves";
  if( cfg.x_label.empty() ) cfg.x_label = "False Positive Rate";
  if( cfg.y_label.empty() ) cfg.y_label = "True Positive Rate";

  d->config = cfg;

  cv::Mat canvas = d->create_canvas();
  d->draw_grid( canvas );

  // Draw diagonal reference line
  cv::Point diag_start = d->data_to_pixel( 0.0, 0.0 );
  cv::Point diag_end = d->data_to_pixel( 1.0, 1.0 );
  cv::line( canvas, diag_start, diag_end,
            cv::Scalar( 180, 180, 180 ), 1, cv::LINE_AA );

  d->draw_axes( canvas );
  d->draw_labels( canvas );

  // Draw each curve
  std::vector< std::pair< std::string, cv::Scalar > > legend_entries;
  size_t color_idx = 0;

  for( const auto& kv : curves )
  {
    const auto& name = kv.first;
    const auto& curve = kv.second;

    std::vector< cv::Point > points;
    for( const auto& pt : curve.points )
    {
      points.push_back( d->data_to_pixel( pt.false_positive_rate, pt.true_positive_rate ) );
    }

    cv::Scalar color = d->config.colors.get_line_color( color_idx++ );
    d->draw_curve( canvas, points, color );

    std::string label = name;
    if( d->config.show_auc && curve.auc > 0 )
    {
      label += " (AUC=" + format_double( curve.auc, 2 ) + ")";
    }
    legend_entries.push_back( { label, color } );
  }

  d->draw_legend( canvas, legend_entries );

  return canvas;
}

// -----------------------------------------------------------------------------
// Confusion Matrix Rendering
// -----------------------------------------------------------------------------

cv::Mat
metrics_plotter::render_confusion_matrix( const confusion_matrix_data& matrix,
                                          bool normalized )
{
  if( matrix.class_names.empty() || matrix.matrix.empty() )
  {
    return cv::Mat();
  }

  size_t num_classes = matrix.class_names.size();

  // Calculate cell size based on number of classes
  int min_cell_size = 40;
  int max_cell_size = 80;
  int cell_size = std::max( min_cell_size,
    std::min( max_cell_size, static_cast< int >( 600 / num_classes ) ) );

  // Calculate canvas size
  int label_margin = 100;
  int top_margin = 60;
  int width = label_margin + static_cast< int >( num_classes ) * cell_size + 50;
  int height = top_margin + static_cast< int >( num_classes ) * cell_size + label_margin;

  cv::Mat canvas( height, width, CV_8UC3, d->config.colors.background );

  // Draw title
  std::string title = d->config.title.empty() ? "Confusion Matrix" : d->config.title;
  int baseline = 0;
  cv::Size title_size = get_text_size( title, priv::font_face,
    d->config.title_font_scale, d->config.font_thickness + 1, &baseline );
  cv::putText( canvas, title,
    cv::Point( ( width - title_size.width ) / 2, 30 ),
    priv::font_face, d->config.title_font_scale, d->config.colors.title_color,
    d->config.font_thickness + 1, cv::LINE_AA );

  // Find max value for normalization
  double max_val = 0.0;
  if( normalized && !matrix.normalized_matrix.empty() )
  {
    for( const auto& row : matrix.normalized_matrix )
    {
      for( double val : row )
      {
        max_val = std::max( max_val, val );
      }
    }
  }
  else
  {
    for( const auto& row : matrix.matrix )
    {
      for( int val : row )
      {
        max_val = std::max( max_val, static_cast< double >( val ) );
      }
    }
  }
  if( max_val == 0.0 ) max_val = 1.0;

  // Draw cells
  for( size_t i = 0; i < num_classes; ++i )
  {
    for( size_t j = 0; j < num_classes; ++j )
    {
      int x = label_margin + static_cast< int >( j ) * cell_size;
      int y = top_margin + static_cast< int >( i ) * cell_size;

      // Get value
      double value = 0.0;
      if( normalized && !matrix.normalized_matrix.empty() &&
          i < matrix.normalized_matrix.size() &&
          j < matrix.normalized_matrix[i].size() )
      {
        value = matrix.normalized_matrix[i][j];
      }
      else if( i < matrix.matrix.size() && j < matrix.matrix[i].size() )
      {
        value = static_cast< double >( matrix.matrix[i][j] );
      }

      // Calculate color
      double t = normalized ? value : ( value / max_val );
      cv::Scalar cell_color = interpolate_color(
        d->config.colors.cm_low, d->config.colors.cm_high, t );

      // Draw cell
      cv::rectangle( canvas,
        cv::Point( x, y ),
        cv::Point( x + cell_size, y + cell_size ),
        cell_color, cv::FILLED );
      cv::rectangle( canvas,
        cv::Point( x, y ),
        cv::Point( x + cell_size, y + cell_size ),
        d->config.colors.grid_color, 1, cv::LINE_AA );

      // Draw value text
      std::string value_str;
      if( normalized )
      {
        value_str = format_double( value, 2 );
      }
      else
      {
        value_str = std::to_string( static_cast< int >( value ) );
      }

      cv::Size text_size = get_text_size( value_str, priv::font_face,
        d->config.font_scale * 0.8, 1, &baseline );

      // Use white text on dark cells, black on light
      cv::Scalar text_color = ( t > 0.5 ) ?
        cv::Scalar( 255, 255, 255 ) : d->config.colors.text_color;

      cv::putText( canvas, value_str,
        cv::Point( x + ( cell_size - text_size.width ) / 2,
                   y + ( cell_size + text_size.height ) / 2 ),
        priv::font_face, d->config.font_scale * 0.8, text_color, 1, cv::LINE_AA );
    }
  }

  // Draw class labels
  double label_font_scale = d->config.font_scale * 0.7;

  // Y-axis labels (true class)
  for( size_t i = 0; i < num_classes; ++i )
  {
    int y = top_margin + static_cast< int >( i ) * cell_size + cell_size / 2;

    std::string label = matrix.class_names[i];
    if( label.length() > 12 )
    {
      label = label.substr( 0, 10 ) + "..";
    }

    cv::Size text_size = get_text_size( label, priv::font_face,
      label_font_scale, 1, &baseline );

    cv::putText( canvas, label,
      cv::Point( label_margin - text_size.width - 5,
                 y + text_size.height / 2 ),
      priv::font_face, label_font_scale, d->config.colors.text_color, 1, cv::LINE_AA );
  }

  // X-axis labels (predicted class) - rotated
  for( size_t j = 0; j < num_classes; ++j )
  {
    int x = label_margin + static_cast< int >( j ) * cell_size + cell_size / 2;
    int y_base = top_margin + static_cast< int >( num_classes ) * cell_size + 5;

    std::string label = matrix.class_names[j];
    if( label.length() > 12 )
    {
      label = label.substr( 0, 10 ) + "..";
    }

    // Draw rotated text (vertical)
    cv::Size text_size = get_text_size( label, priv::font_face,
      label_font_scale, 1, &baseline );

    cv::Mat text_img( text_size.width + 10, text_size.height + 10, CV_8UC3,
                      d->config.colors.background );
    cv::putText( text_img, label, cv::Point( 5, text_size.height + 2 ),
      priv::font_face, label_font_scale, d->config.colors.text_color, 1, cv::LINE_AA );

    cv::Mat rotated;
    cv::rotate( text_img, rotated, cv::ROTATE_90_COUNTERCLOCKWISE );

    int dest_x = x - rotated.cols / 2;
    int dest_y = y_base;

    if( dest_x >= 0 && dest_x + rotated.cols <= canvas.cols &&
        dest_y >= 0 && dest_y + rotated.rows <= canvas.rows )
    {
      rotated.copyTo( canvas( cv::Rect( dest_x, dest_y, rotated.cols, rotated.rows ) ) );
    }
  }

  // Axis titles
  cv::putText( canvas, "Predicted",
    cv::Point( label_margin + static_cast< int >( num_classes ) * cell_size / 2 - 30,
               height - 10 ),
    priv::font_face, d->config.font_scale, d->config.colors.text_color, 1, cv::LINE_AA );

  return canvas;
}

// -----------------------------------------------------------------------------
// Histogram Rendering
// -----------------------------------------------------------------------------

cv::Mat
metrics_plotter::render_histogram( const std::vector< int >& bins,
                                   const std::vector< std::string >& bin_labels )
{
  if( bins.empty() )
  {
    return cv::Mat();
  }

  plot_config cfg = d->config;
  cfg.y_min = 0;
  cfg.y_max = *std::max_element( bins.begin(), bins.end() ) * 1.1;
  cfg.x_min = 0;
  cfg.x_max = static_cast< double >( bins.size() );
  cfg.num_x_ticks = std::min( static_cast< int >( bins.size() ) + 1, 11 );
  cfg.show_grid = true;

  d->config = cfg;

  cv::Mat canvas = d->create_canvas();
  d->draw_grid( canvas );
  d->draw_axes( canvas );
  d->draw_labels( canvas );

  // Calculate bar width
  int bar_width = d->plot_width() / static_cast< int >( bins.size() ) - 2;
  bar_width = std::max( bar_width, 5 );

  cv::Scalar bar_color = d->config.colors.get_line_color( 0 );

  for( size_t i = 0; i < bins.size(); ++i )
  {
    double x = static_cast< double >( i ) + 0.5;
    cv::Point top = d->data_to_pixel( x, static_cast< double >( bins[i] ) );
    cv::Point bottom = d->data_to_pixel( x, 0 );

    cv::rectangle( canvas,
      cv::Point( top.x - bar_width / 2, top.y ),
      cv::Point( top.x + bar_width / 2, bottom.y ),
      bar_color, cv::FILLED );
    cv::rectangle( canvas,
      cv::Point( top.x - bar_width / 2, top.y ),
      cv::Point( top.x + bar_width / 2, bottom.y ),
      d->config.colors.axis_color, 1, cv::LINE_AA );
  }

  // Draw bin labels if provided
  if( !bin_labels.empty() )
  {
    for( size_t i = 0; i < std::min( bins.size(), bin_labels.size() ); ++i )
    {
      double x = static_cast< double >( i ) + 0.5;
      cv::Point pt = d->data_to_pixel( x, 0 );

      int baseline = 0;
      cv::Size text_size = get_text_size( bin_labels[i], priv::font_face,
        d->config.font_scale * 0.7, 1, &baseline );

      cv::putText( canvas, bin_labels[i],
        cv::Point( pt.x - text_size.width / 2, d->plot_bottom() + 35 ),
        priv::font_face, d->config.font_scale * 0.7,
        d->config.colors.text_color, 1, cv::LINE_AA );
    }
  }

  return canvas;
}

cv::Mat
metrics_plotter::render_histogram( const std::map< int, int >& data )
{
  if( data.empty() )
  {
    return cv::Mat();
  }

  std::vector< int > bins;
  std::vector< std::string > labels;

  for( const auto& kv : data )
  {
    bins.push_back( kv.second );
    labels.push_back( std::to_string( kv.first ) );
  }

  return render_histogram( bins, labels );
}

// -----------------------------------------------------------------------------
// Combined Plot Rendering
// -----------------------------------------------------------------------------

bool
metrics_plotter::render_all_plots( const evaluation_plot_data& plot_data,
                                   const std::string& output_dir )
{
  bool success = true;

  // Create output directory if needed
  // (caller should ensure directory exists)

  // Overall PR curve
  if( !plot_data.overall_pr_curve.points.empty() )
  {
    plot_config cfg;
    cfg.title = "Overall Precision-Recall Curve";
    set_config( cfg );

    cv::Mat pr_img = render_pr_curve( plot_data.overall_pr_curve );
    success = save_plot( pr_img, output_dir + "/pr_curve_overall.png" ) && success;
  }

  // Per-class PR curves
  if( !plot_data.per_class_pr_curves.empty() )
  {
    plot_config cfg;
    cfg.title = "Per-Class Precision-Recall Curves";
    set_config( cfg );

    cv::Mat pr_img = render_pr_curves( plot_data.per_class_pr_curves );
    success = save_plot( pr_img, output_dir + "/pr_curves_per_class.png" ) && success;
  }

  // Overall ROC curve
  if( !plot_data.overall_roc_curve.points.empty() )
  {
    plot_config cfg;
    cfg.title = "Overall ROC Curve";
    set_config( cfg );

    cv::Mat roc_img = render_roc_curve( plot_data.overall_roc_curve );
    success = save_plot( roc_img, output_dir + "/roc_curve_overall.png" ) && success;
  }

  // Confusion matrix
  if( !plot_data.confusion_matrix.class_names.empty() )
  {
    plot_config cfg;
    cfg.title = "Confusion Matrix";
    set_config( cfg );

    cv::Mat conf_img = render_confusion_matrix( plot_data.confusion_matrix );
    success = save_plot( conf_img, output_dir + "/confusion_matrix.png" ) && success;
  }

  // IoU histogram
  if( !plot_data.iou_histogram.empty() )
  {
    plot_config cfg;
    cfg.title = "IoU Distribution";
    cfg.x_label = "IoU";
    cfg.y_label = "Count";
    set_config( cfg );

    std::vector< std::string > labels;
    for( size_t i = 0; i < plot_data.iou_histogram.size(); ++i )
    {
      labels.push_back( format_double( i * 0.05, 2 ) );
    }

    cv::Mat hist_img = render_histogram( plot_data.iou_histogram, labels );
    success = save_plot( hist_img, output_dir + "/iou_histogram.png" ) && success;
  }

  // Track purity histogram
  if( !plot_data.track_purity_histogram.empty() )
  {
    plot_config cfg;
    cfg.title = "Track Purity Distribution";
    cfg.x_label = "Purity %";
    cfg.y_label = "Count";
    set_config( cfg );

    std::vector< std::string > labels = {
      "0-10", "10-20", "20-30", "30-40", "40-50",
      "50-60", "60-70", "70-80", "80-90", "90-100"
    };

    cv::Mat hist_img = render_histogram( plot_data.track_purity_histogram, labels );
    success = save_plot( hist_img, output_dir + "/track_purity_histogram.png" ) && success;
  }

  // Track continuity histogram
  if( !plot_data.track_continuity_histogram.empty() )
  {
    plot_config cfg;
    cfg.title = "Track Continuity Distribution";
    cfg.x_label = "Continuity %";
    cfg.y_label = "Count";
    set_config( cfg );

    std::vector< std::string > labels = {
      "0-10", "10-20", "20-30", "30-40", "40-50",
      "50-60", "60-70", "70-80", "80-90", "90-100"
    };

    cv::Mat hist_img = render_histogram( plot_data.track_continuity_histogram, labels );
    success = save_plot( hist_img, output_dir + "/track_continuity_histogram.png" ) && success;
  }

  // Track length histogram
  if( !plot_data.track_length_histogram.empty() )
  {
    plot_config cfg;
    cfg.title = "Track Length Distribution";
    cfg.x_label = "Length (frames)";
    cfg.y_label = "Count";
    set_config( cfg );

    cv::Mat hist_img = render_histogram( plot_data.track_length_histogram );
    success = save_plot( hist_img, output_dir + "/track_length_histogram.png" ) && success;
  }

  return success;
}

// -----------------------------------------------------------------------------
// Utility Functions
// -----------------------------------------------------------------------------

bool
metrics_plotter::save_plot( const cv::Mat& image, const std::string& filepath )
{
  if( image.empty() )
  {
    return false;
  }

  try
  {
    return cv::imwrite( filepath, image );
  }
  catch( const cv::Exception& e )
  {
    return false;
  }
}

} // namespace viame
