
import unittest
import sys
import os
import numpy as np

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from calculate_weights import calculate_priority_score

class TestCalculatePriorityScore(unittest.TestCase):

    def test_example_case(self):
        """제공된 예시와 동일한 조건으로 우선순위 점수를 계산합니다."""
        score = calculate_priority_score(
            damage_severity=4,           # 파손 심각도 4점
            complaint_ratio=1.8,         # 평균 대비 1.8배 민원
            traffic_ratios=[2.5, 3.0],   # 인근 교차로 2곳의 차량 비율
            weights=(0.5, 0.25, 0.25)
        )
        
        # 예상 결과 계산
        # A = 4
        # B_capped = min(1.8, 2.5) = 1.8
        # B_scaled = 1 + 4 * (1.8 / 2.5) = 1 + 4 * 0.72 = 1 + 2.88 = 3.88
        # C_raw = mean([2.5, 3.0]) = 2.75
        # C_capped = min(2.75, 3.21) = 2.75
        # C_scaled = 1 + 4 * (2.75 / 3.21) = 1 + 4 * 0.856... = 1 + 3.426... = 4.426...
        # final_score = 0.5 * 4 + 0.25 * 3.88 + 0.25 * 4.426...
        #             = 2 + 0.97 + 1.106... = 4.076...
        
        # 예상 점수와 실제 점수가 매우 근사한지 확인
        self.assertAlmostEqual(score, 4.0766978, places=5)

    def test_scaling_boundaries(self):
        """스케일링 함수의 경계값을 테스트합니다."""
        # B 스케일링 테스트
        # B_raw = 0 -> B_scaled = 1
        score_b_min = calculate_priority_score(1, 0, [], (0, 1, 0))
        self.assertEqual(score_b_min, 1.0)
        
        # B_raw = 2.5 -> B_scaled = 5
        score_b_max = calculate_priority_score(1, 2.5, [], (0, 1, 0))
        self.assertEqual(score_b_max, 5.0)

        # B_raw > 2.5 (capped) -> B_scaled = 5
        score_b_over = calculate_priority_score(1, 3.0, [], (0, 1, 0))
        self.assertEqual(score_b_over, 5.0)

        # C 스케일링 테스트
        # C_raw = 0 -> C_scaled = 1
        score_c_min = calculate_priority_score(1, 0, [0], (0, 0, 1))
        self.assertEqual(score_c_min, 1.0)

        # C_raw = 3.21 -> C_scaled = 5
        score_c_max = calculate_priority_score(1, 0, [3.21], (0, 0, 1))
        self.assertEqual(score_c_max, 5.0)

        # C_raw > 3.21 (capped) -> C_scaled = 5
        score_c_over = calculate_priority_score(1, 0, [4.0], (0, 0, 1))
        self.assertEqual(score_c_over, 5.0)

    def test_no_traffic_data(self):
        """교차로 데이터가 없을 때 C_scaled가 1.0이 되는지 확인합니다."""
        score = calculate_priority_score(
            damage_severity=3,
            complaint_ratio=1.5,
            traffic_ratios=[], # 빈 리스트
            weights=(0.4, 0.3, 0.3)
        )
        
        # A = 3
        # B_scaled = 1 + 4 * (1.5 / 2.5) = 3.4
        # C_scaled = 1.0
        # final_score = 0.4 * 3 + 0.3 * 3.4 + 0.3 * 1.0 = 1.2 + 1.02 + 0.3 = 2.52
        self.assertAlmostEqual(score, 2.52, places=5)

if __name__ == '__main__':
    unittest.main()
