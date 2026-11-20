#!/usr/bin/env python3
"""Isaac 미션 — 신호등 · 시작 바코드는 성공한 것으로 치고 SURVEY(주행 · 퍽 집기)부터 한 판.

  python3 isaac/mission/run_from_survey.py                               # R5 Y3 G4 · P2-01
  python3 isaac/mission/run_from_survey.py --counts R4Y4G4 --order desc  # 횟수 · 내림차순
  python3 isaac/mission/run_from_survey.py --clear-basket                # 적재함 장부가 남아 시작이 거부될 때

옵션은 run_mission.py와 같고 --from-survey만 늘 켠다. mission_manager가 start:FROM_SURVEY:<바코드>:<횟수>를 받아
GO_OBSERVE · BARCODE · READ를 건너뛰고 SURVEY → BUILD → VERIFY → RETURN.
"""
import sys

from run_mission import main

if __name__ == '__main__':
    sys.argv.append('--from-survey')
    main()
