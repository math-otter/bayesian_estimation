import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta

# 한글 폰트 깨지지 않게 설정
# plt.rc("font", family="Nanum Gothic") # Mac이나 리눅스에서 사용할 수 있음
plt.rc("font", family="Malgun Gothic")  # Windows에서 사용

######## 가챠와 데이터 수집
# 가챠 시뮬레이션 설정
np.random.seed(42)  # 랜덤 시드 고정
p = 0.5
t = 20
x = np.random.binomial(1, p, t) # t개의 난수 생성: 0.1의 확률로 1, 0.9의 확률로 0

# 데이터 프레임으로 결과 저장
df = pd.DataFrame({
    "시도": np.arange(1, t + 1),
    "성공 여부": x
})

######## 베이지안 추정을 통한 검증
# 베이지안 업데이트 함수 정의
def calculate_posterior(successes, failures, alpha_prior, beta_prior):
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures
    return alpha_post, beta_post

# 사전 경험 반영
alpha_prior = 1
beta_prior = 1

# 누적 성공 횟수와 실패 횟수 계산
df["누적 성공 횟수"] = df["성공 여부"].cumsum()
df["누적 실패 횟수"] = (df["시도"] - df["누적 성공 횟수"])

# 매 시도에서 데이터를 반영하여 사후 분포 계산 및 그리기
p_values = np.linspace(0, 1, 100) # 시각화를 위해 0~1 사이의 값을 갖는 P값 생성

beta_pdf_prior = beta.pdf(p_values, alpha_prior, beta_prior)
plt.plot(p_values, beta_pdf_prior, label=f"사전 분포: Beta({alpha_prior}, {beta_prior})", color="gray", lw=2, linestyle="--")

for i, row in df.iterrows():
    successes = row["누적 성공 횟수"]
    failures = row["누적 실패 횟수"]
    alpha_post, beta_post = calculate_posterior(successes, failures, alpha_prior, beta_prior)    
    beta_pdf_post = beta.pdf(p_values, alpha_post, beta_post)
    opacity = (i + 1) / len(df)    
    plt.plot(p_values, beta_pdf_post, 
             label=f"사후 분포 {int(row['시도'])}: Beta({alpha_post}, {beta_post})", 
             lw=1, color="black", alpha=opacity)

final_alpha_post = alpha_post
final_beta_post = beta_post
expected_value = final_alpha_post / (final_alpha_post + final_beta_post)
plt.axvline(expected_value, color="red", linestyle='-.', lw=2, label=f"최종 분포의 기댓값: {expected_value:.3f}")

# 시각화
plt.title("각 시도에서의 사후 분포", fontsize=16)
plt.xlabel("성공 확률", fontsize=12)
plt.ylabel("성공 확률의 확률 밀도", fontsize=12)
plt.legend(loc="upper right", fontsize=10)
plt.grid(True)
plt.show()