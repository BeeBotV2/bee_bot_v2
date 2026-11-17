#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "nav_msgs/msg/path.hpp"

namespace ml {

// Shortest distance from a pose to the path polyline, in the path's frame.
// Returns NaN when the path has no usable segment or the pose is invalid.
inline double cross_track_error(double x, double y, const nav_msgs::msg::Path & path) {
  if (!std::isfinite(x) || !std::isfinite(y) || path.poses.size() < 2) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double best_sq = std::numeric_limits<double>::infinity();
  for (std::size_t i = 1; i < path.poses.size(); ++i) {
    const auto & a = path.poses[i - 1].pose.position;
    const auto & b = path.poses[i].pose.position;
    if (!std::isfinite(a.x) || !std::isfinite(a.y) || !std::isfinite(b.x) || !std::isfinite(b.y)) {
      continue;
    }

    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    const double length_sq = dx * dx + dy * dy;
    if (length_sq <= 1e-12) continue;

    const double projection = std::clamp(((x - a.x) * dx + (y - a.y) * dy) / length_sq, 0.0, 1.0);
    const double ex = x - (a.x + projection * dx);
    const double ey = y - (a.y + projection * dy);
    best_sq = std::min(best_sq, ex * ex + ey * ey);
  }
  return std::isfinite(best_sq) ? std::sqrt(best_sq) : std::numeric_limits<double>::quiet_NaN();
}

}  // namespace ml
