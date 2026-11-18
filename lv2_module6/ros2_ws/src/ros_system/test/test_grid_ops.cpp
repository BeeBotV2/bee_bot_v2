// nvblox_map C++판 격자 계산 (include/ros_system/grid_ops.hpp) — scripts/nvblox_map.py despeckle · overlay와 같은 뜻인지
#include <gtest/gtest.h>

#include <vector>

#include "ros_system/grid_ops.hpp"

using nav_msgs::msg::OccupancyGrid;

static OccupancyGrid make(unsigned w, unsigned h, std::vector<int8_t> data, double ox = 0.0, double oy = 0.0) {
  OccupancyGrid g;
  g.info.width = w;
  g.info.height = h;
  g.info.resolution = 0.05;
  g.info.origin.position.x = ox;
  g.info.origin.position.y = oy;
  g.data = std::move(data);
  return g;
}

TEST(GridOps, PgmTopRowIsHighestYAndTrinary) {
  // 2×2: 아래 줄(y=0) [점유 100, 빈칸 0] · 위 줄(y=1) [모름 -1, 애매 50] → pgm은 위 줄부터
  const auto g = make(2, 2, {100, 0, -1, 50}, -1.5, 2.0);
  const std::string pgm = ros_system::to_pgm(g);
  const std::string head = "P5\n2 2\n255\n";
  ASSERT_EQ(pgm.size(), head.size() + 4);
  EXPECT_EQ(pgm.substr(0, head.size()), head);
  const auto px = [&](size_t i) { return static_cast<unsigned char>(pgm[head.size() + i]); };
  EXPECT_EQ(px(0), 205);  // 위 왼쪽 모름
  EXPECT_EQ(px(1), 205);  // 위 오른쪽 50 = 0.25~0.65 사이 → 모름
  EXPECT_EQ(px(2), 0);    // 아래 왼쪽 점유
  EXPECT_EQ(px(3), 254);  // 아래 오른쪽 빈칸
  const std::string yaml = ros_system::to_map_yaml(g, "map.pgm");
  EXPECT_NE(yaml.find("image: map.pgm"), std::string::npos);
  EXPECT_NE(yaml.find("resolution: 0.05"), std::string::npos);
  EXPECT_NE(yaml.find("origin: [-1.5, 2, 0]"), std::string::npos);
}

TEST(GridOps, DespeckleRemovesSmallBlobsEightConnected) {
  // 5×4: 왼쪽 위 대각선 2칸(8방향 이웃 = 한 덩어리) · 오른쪽 아래 3칸 덩어리 · 49는 점유 아님
  const auto g = make(5, 4, {100, 0, 0, 0, 0,
                             0, 60, 0, 49, 0,
                             0, 0, 0, 100, 100,
                             -1, 0, 0, 0, 100});
  long n = 0;
  auto out = ros_system::despeckle(g, 3, n);
  EXPECT_EQ(n, 2);
  EXPECT_EQ(out.data[0], -1);
  EXPECT_EQ(out.data[6], -1);
  EXPECT_EQ(out.data[8], 49);
  EXPECT_EQ(out.data[13], 100);
  EXPECT_EQ(out.data[19], 100);
  EXPECT_EQ(out.data[15], -1);  // 원래 모름은 그대로
  out = ros_system::despeckle(g, 2, n);  // 2칸 덩어리는 2보다 작지 않음
  EXPECT_EQ(n, 0);
  EXPECT_EQ(out.data, g.data);
}

TEST(GridOps, OverlayKeepsWallsAndClipsOutside) {
  const auto walls = make(4, 3, {100, 0, 0, 0,
                                 0, 0, 100, 0,
                                 0, 0, 0, 0}, -0.1, -0.05);
  // nvblox 격자 원점이 벽 격자에서 (+1, +1)칸 · 3×3 중 오른쪽·위쪽 한 줄은 밖으로 나감
  const auto grid = make(4, 3, {10, 20, 30, 40,
                                50, 60, 70, 80,
                                90, 91, 92, 93}, -0.05, 0.0);
  auto out = ros_system::overlay(walls, &grid);
  const std::vector<int8_t> want = {100, 0, 0, 0,
                                    0, 10, 100, 30,
                                    0, 50, 60, 70};
  EXPECT_EQ(out.data, want);
  EXPECT_EQ(out.info.width, 4u);
  EXPECT_EQ(ros_system::overlay(walls, nullptr).data, walls.data);
  const auto far = make(1, 1, {55}, 10.0, 10.0);
  EXPECT_EQ(ros_system::overlay(walls, &far).data, walls.data);
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
