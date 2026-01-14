/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Plotting utilities for evaluation metrics using OpenCV

#ifndef VIAME_OPENCV_PLOT_METRICS_H
#define VIAME_OPENCV_PLOT_METRICS_H

#include "viame_opencv_export.h"

#include <viame/evaluate_models.h>

#include <opencv2/core/core.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace viame {

// ----------------------------------------------------------------------------
/// \brief Color scheme for plots
struct VIAME_OPENCV_EXPORT plot_color_scheme
{
  cv::Scalar background = cv::Scalar( 255, 255, 255 );      // White
  cv::Scalar plot_area = cv::Scalar( 250, 250, 250 );       // Light gray
  cv::Scalar axis_color = cv::Scalar( 60, 60, 60 );         // Dark gray
  cv::Scalar grid_color = cv::Scalar( 220, 220, 220 );      // Light gray
  cv::Scalar text_color = cv::Scalar( 40, 40, 40 );         // Near black
  cv::Scalar title_color = cv::Scalar( 20, 20, 20 );        // Black

  // Line colors for multiple curves (BGR format)
  std::vector< cv::Scalar > line_colors = {
    cv::Scalar( 91, 35, 37 ),     // Dark blue
    cv::Scalar( 106, 78, 145 ),   // Purple
    cv::Scalar( 66, 133, 43 ),    // Green
    cv::Scalar( 43, 87, 191 ),    // Orange-red
    cv::Scalar( 180, 119, 31 ),   // Teal
    cv::Scalar( 32, 165, 218 ),   // Gold
    cv::Scalar( 128, 64, 128 ),   // Magenta
    cv::Scalar( 64, 128, 128 ),   // Olive
  };

  // Confusion matrix colors (low to high)
  cv::Scalar cm_low = cv::Scalar( 255, 255, 255 );          // White
  cv::Scalar cm_high = cv::Scalar( 120, 50, 20 );           // Dark blue

  /// Get line color for index (cycles through available colors)
  cv::Scalar get_line_color( size_t index ) const;
};

// ----------------------------------------------------------------------------
/// \brief Configuration options for plot rendering
struct VIAME_OPENCV_EXPORT plot_config
{
  // Canvas size
  int width = 800;
  int height = 600;

  // Margins (pixels)
  int margin_left = 80;
  int margin_right = 40;
  int margin_top = 60;
  int margin_bottom = 60;

  // Title and labels
  std::string title;
  std::string x_label;
  std::string y_label;

  // Axis ranges (auto-computed if min >= max)
  double x_min = 0.0;
  double x_max = 1.0;
  double y_min = 0.0;
  double y_max = 1.0;

  // Grid and ticks
  bool show_grid = true;
  int num_x_ticks = 11;   // For 0.0, 0.1, ..., 1.0
  int num_y_ticks = 11;

  // Line appearance
  int line_thickness = 2;
  bool show_points = false;
  int point_radius = 3;

  // Legend
  bool show_legend = true;
  int legend_position = 0;  // 0=auto, 1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right

  // Font settings
  double font_scale = 0.5;
  double title_font_scale = 0.7;
  int font_thickness = 1;

  // Color scheme
  plot_color_scheme colors;

  // Annotation options
  bool show_auc = true;           // Show AUC/AP value on plot
  bool show_best_threshold = true; // Show best operating point
};

// ----------------------------------------------------------------------------
/// \brief Main class for rendering evaluation metric plots
///
/// This class provides methods to render various evaluation plots using OpenCV,
/// including precision-recall curves, ROC curves, confusion matrices, and
/// histogram plots.
///
/// Example usage:
/// \code
/// viame::metrics_plotter plotter;
/// plotter.set_config(config);
///
/// cv::Mat pr_image = plotter.render_pr_curve(pr_curve_data);
/// cv::imwrite("pr_curve.png", pr_image);
///
/// cv::Mat conf_image = plotter.render_confusion_matrix(conf_matrix_data);
/// cv::imwrite("confusion_matrix.png", conf_image);
/// \endcode
class VIAME_OPENCV_EXPORT metrics_plotter
{
public:
  /// Constructor
  metrics_plotter();

  /// Destructor
  ~metrics_plotter();

  /// Copy constructor (deleted)
  metrics_plotter( const metrics_plotter& ) = delete;

  /// Move constructor
  metrics_plotter( metrics_plotter&& ) noexcept;

  /// Copy assignment (deleted)
  metrics_plotter& operator=( const metrics_plotter& ) = delete;

  /// Move assignment
  metrics_plotter& operator=( metrics_plotter&& ) noexcept;

  /// \brief Set the plot configuration
  /// \param config Configuration options
  void set_config( const plot_config& config );

  /// \brief Get the current plot configuration
  /// \returns Current configuration
  plot_config get_config() const;

  // -------------------------------------------------------------------------
  // Precision-Recall Curve Rendering
  // -------------------------------------------------------------------------

  /// \brief Render a single precision-recall curve
  /// \param curve The PR curve data to render
  /// \returns OpenCV Mat containing the rendered plot
  cv::Mat render_pr_curve( const pr_curve_data& curve );

  /// \brief Render multiple precision-recall curves on the same plot
  /// \param curves Map of curve name to PR curve data
  /// \returns OpenCV Mat containing the rendered plot
  cv::Mat render_pr_curves( const std::map< std::string, pr_curve_data >& curves );

  // -------------------------------------------------------------------------
  // ROC Curve Rendering
  // -------------------------------------------------------------------------

  /// \brief Render a single ROC curve
  /// \param curve The ROC curve data to render
  /// \returns OpenCV Mat containing the rendered plot
  cv::Mat render_roc_curve( const roc_curve_data& curve );

  /// \brief Render multiple ROC curves on the same plot
  /// \param curves Map of curve name to ROC curve data
  /// \returns OpenCV Mat containing the rendered plot
  cv::Mat render_roc_curves( const std::map< std::string, roc_curve_data >& curves );

  // -------------------------------------------------------------------------
  // Confusion Matrix Rendering
  // -------------------------------------------------------------------------

  /// \brief Render a confusion matrix as a heatmap
  /// \param matrix The confusion matrix data to render
  /// \param normalized If true, use normalized values (default: true)
  /// \returns OpenCV Mat containing the rendered plot
  cv::Mat render_confusion_matrix( const confusion_matrix_data& matrix,
                                   bool normalized = true );

  // -------------------------------------------------------------------------
  // Histogram Rendering
  // -------------------------------------------------------------------------

  /// \brief Render a histogram from bin data
  /// \param bins Vector of bin counts
  /// \param bin_labels Labels for each bin (optional)
  /// \returns OpenCV Mat containing the rendered plot
  cv::Mat render_histogram( const std::vector< int >& bins,
                            const std::vector< std::string >& bin_labels = {} );

  /// \brief Render a histogram from a map (e.g., track length distribution)
  /// \param data Map of value to count
  /// \returns OpenCV Mat containing the rendered plot
  cv::Mat render_histogram( const std::map< int, int >& data );

  // -------------------------------------------------------------------------
  // Combined Plot Rendering
  // -------------------------------------------------------------------------

  /// \brief Render all plots from evaluation plot data
  /// \param plot_data The complete plot data from evaluation
  /// \param output_dir Directory to save plot images
  /// \returns true on success
  bool render_all_plots( const evaluation_plot_data& plot_data,
                         const std::string& output_dir );

  // -------------------------------------------------------------------------
  // Utility Functions
  // -------------------------------------------------------------------------

  /// \brief Save a rendered plot to file
  /// \param image The rendered plot image
  /// \param filepath Output file path (PNG, JPG, etc.)
  /// \returns true on success
  static bool save_plot( const cv::Mat& image, const std::string& filepath );

private:
  class priv;
  std::unique_ptr< priv > d;
};

} // namespace viame

#endif // VIAME_OPENCV_PLOT_METRICS_H
