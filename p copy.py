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

# 관측된 데이터
t_A = 10
x_A = 1
t_B = 3
x_B = 3

# 사전 분포와 사후 분포
p_values = np.linspace(0, 1, 100) # 0~1 사이의 값을 갖는 P값 생성

beta_pdf_A_pri = beta.pdf(p_values, alpha_A, beta_A)
beta_pdf_B_pri = beta.pdf(p_values, alpha_B, beta_B)

beta_pdf_A_pos = beta.pdf(p_values, alpha_A + x_A, beta_A + (t_A - x_A))
beta_pdf_B_pos = beta.pdf(p_values, alpha_B + x_B, beta_B + (t_B - x_B))


# 시각화
plt.plot(p_values, beta_pdf_A_pri, label=f"A_pri: Beta({alpha_A}, {beta_A})", color="blue", alpha=0.5, lw=2)
plt.plot(p_values, beta_pdf_A_pos, label=f"A_pos: Beta({alpha_A}+{x_A}, {beta_A}+{t_A - x_A})", color="blue", lw=2)
plt.plot(p_values, beta_pdf_B_pri, label=f"B_pri: Beta({alpha_B}, {beta_B})", color="green", alpha=0.5, lw=2, linestyle="--")
plt.plot(p_values, beta_pdf_B_pos, label=f"B_pos: Beta({alpha_B}+{x_B}, {beta_B}+{t_B - x_B})", color="green", lw=2, linestyle="--")

plt.title("데이터 관측 전후 A, B 회사의 성공 확률 비교", fontsize=14)
plt.xlabel("성공 확률", fontsize=12)
plt.ylabel("확률밀도", fontsize=12)
plt.legend()
plt.grid()
plt.savefig(r"Images\A, B 회사의 성공 확률 비교")
