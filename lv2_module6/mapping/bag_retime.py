#!/usr/bin/env python3
"""bag 시각을 메시지 header.stamp로 다시 씀 — Isaac에서 --use-sim-time 없이 녹화한 bag 살리기.

  python3 mapping/bag_retime.py ~/.ros/beebot_bags/lap1 ~/.ros/beebot_bags/lap1_sim

왜: 녹화 때 bag 시각 = PC 실제 시각인데 Isaac 메시지 header는 시뮬 시계. 재생(ros2 bag play --clock)이 PC 시각을 /clock으로 내면
  EKF · TF가 header 시각과 어긋나 odom이 튀고 cuVSLAM이 map을 못 냄 (11-19 lap1: /clock 1795014476 · header 71 s).
어떻게: zstd 파일 압축을 스트림으로 풀며(디스크에 7 GB 풀린 파일을 안 만듦) 메시지마다 CDR 앞 12바이트의 header.stamp를 읽어
  log · publish 시각으로 다시 씀 (zstd 청크 압축). 녹화 토픽은 전부 std_msgs/Header로 시작해야 한다 (mapping.sh BAG_TOPICS).
끝나면 `ros2 bag reindex`로 metadata.yaml을 만든다. 원본은 그대로 둔다.
"""
import glob
import os
import struct
import subprocess
import sys

from mcap.stream_reader import StreamReader
from mcap.records import Channel, Message, Schema
from mcap.writer import CompressionType, Writer


def header_ns(data):
    """CDR 메시지 맨 앞 std_msgs/Header의 stamp → ns (encapsulation 4바이트 뒤 sec int32 · nanosec uint32)"""
    fmt = '<iI' if data[1] == 1 else '>iI'  # 1 = CDR little endian
    sec, nsec = struct.unpack_from(fmt, data, 4)
    return sec * 1_000_000_000 + nsec


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = (os.path.expanduser(a) for a in sys.argv[1:])
    # 같은 이름의 .mcap · .mcap.zstd가 둘 다 있으면 하나만 — ros2 bag play가 압축 bag 옆에 풀어 둔 .mcap을 남김 (둘 다 읽으면 메시지가 두 배)
    zst = glob.glob(os.path.join(src, '*.mcap.zstd'))
    files = sorted(zst + [f for f in glob.glob(os.path.join(src, '*.mcap')) if f + '.zstd' not in zst])
    if not files:
        sys.exit(f'mcap 없음: {src}')
    if os.path.exists(dst):
        sys.exit(f'이미 있음: {dst}')
    os.makedirs(dst)
    out = open(os.path.join(dst, os.path.basename(files[0]).replace('.zstd', '')), 'wb')
    w = Writer(out, compression=CompressionType.ZSTD)
    w.start(profile='ros2', library='bag_retime')
    schemas, channels, n, first, last = {}, {}, 0, None, None
    for f in files:
        dec = subprocess.Popen(['zstd', '-dc', f], stdout=subprocess.PIPE) if f.endswith('.zstd') else None
        stream = dec.stdout if dec else open(f, 'rb')
        for r in StreamReader(stream, skip_magic=False).records:
            if isinstance(r, Schema) and r.id not in schemas:
                schemas[r.id] = w.register_schema(r.name, r.encoding, r.data)
            elif isinstance(r, Channel) and r.id not in channels:
                channels[r.id] = w.register_channel(r.topic, r.message_encoding, schemas[r.schema_id], r.metadata)
            elif isinstance(r, Message):
                t = header_ns(r.data)
                w.add_message(channels[r.channel_id], log_time=t, data=r.data, publish_time=t, sequence=r.sequence)
                n += 1
                first = t if first is None else min(first, t)
                last = t if last is None else max(last, t)
                if n % 10000 == 0:
                    print(f'  {n} 메시지 …', flush=True)
        if dec and dec.wait() != 0:
            sys.exit(f'zstd 풀기 실패: {f}')
    w.finish()
    out.close()
    print(f'{n} 메시지 · header 시각 {first / 1e9:.1f} ~ {last / 1e9:.1f} s → {dst}')
    subprocess.run(['ros2', 'bag', 'reindex', dst, '-s', 'mcap'], check=True)


if __name__ == '__main__':
    main()
