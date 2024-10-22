import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# 한글 폰트 깨지지 않게 설정
# plt.rc("font", family="Nanum Gothic") # Mac이나 리눅스에서 사용할 수 있음
plt.rc("font", family="Malgun Gothic")  # Windows에서 사용

# 베타분포의 파라미터 정의
alpha_A = 4  # 회사 A
beta_A = 3
alpha_B = 1  # 회사 B
beta_B = 1

# 베타분포 생성
p_values = np.linspace(0, 1, 100) # 0~1 사이의 값을 갖는 P값 생성
beta_pdf_A = beta.pdf(p_values, alpha_A, beta_A)
beta_pdf_B = beta.pdf(p_values, alpha_B, beta_B)

# 시각화
plt.plot(p_values, beta_pdf_A, label=f"A: Beta({alpha_A}, {beta_A})", color="blue", lw=2)
plt.plot(p_values, beta_pdf_B, label=f"B: Beta({alpha_B}, {beta_B})", color="green", lw=2, linestyle="--")
plt.title("A, B 회사의 성공 확률 비교", fontsize=14)
plt.xlabel("성공 확률", fontsize=12)
plt.ylabel("확률밀도", fontsize=12)
plt.legend()
plt.grid()
plt.savefig(r"Images\A, B 회사의 성공 확률 비교")
