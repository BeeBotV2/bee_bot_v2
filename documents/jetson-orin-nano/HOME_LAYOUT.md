# Jetson source and data layout

`/home/bee/Lv2_bee_Assignment` is the canonical checkout on branch `develop`. It now contains the current upstream `develop`, the latest `sjh/feature/isaac-setup` changes, the 2D-map converter, and the reviewed changes from the `ghj/perf/jetson-gpu-offload` working tree.

The secondary checkouts remain in place while the robot is still launched from `ghj`. Do not remove or rename `/home/ghj/Lv2_bee_Assignment` until runtime processes have been stopped and relaunched from `bee`; `ghj` still has uncommitted tuning and its installed ROS workspace. `/home/sjh/Lv2_bee_Assignment` contains its Python environment and Isaac experiment records. The reproducible Isaac loop `install`, `log`, `build`, and `__pycache__` outputs were removed; `real_logs` and bag files were preserved.

The 8 GB swap-independent ROS build cache under the old `bee/ros2_ws/build` and its colcon `log` were removed after confirming their generated markers and no active build process. `install/` was preserved. Rebuild from the canonical checkout while the robot is stopped before launching it.

No separate temporary SSD is mounted on Jetson. All `/home` data resides on the root NVMe. Robot MCAP bags, real experiment logs, and Python virtual environments were kept because they contain diagnostic evidence or runtime dependencies.
