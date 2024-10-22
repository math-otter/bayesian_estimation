import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
import imageio.v2 as imageio
import os

# 한글 폰트 깨지지 않게 설정
# plt.rc("font", family="Nanum Gothic") # Mac이나 리눅스에서 사용할 수 있음
plt.rc("font", family="Malgun Gothic")  # Windows에서 사용

######## 가챠와 데이터 수집 ########
# 가챠 시뮬레이션 설정
np.random.seed(42)  # 랜덤 시드 고정
p = 0.5
t = 5
x = np.random.binomial(1, p, t) # t개의 난수 생성: 0.1의 확률로 1, 0.9의 확률로 0

# 데이터 프레임으로 결과 저장
df = pd.DataFrame({
    "시도": np.arange(1, t + 1),
    "성공 여부": x
})

######## 베이지안 추정을 통한 검증 ########
# 베이지안 업데이트 함수 정의
def calculate_posterior(successes, failures, alpha_prior, beta_prior):
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures
    return alpha_post, beta_post

# 사전 경험 반영
alpha_prior = 1
beta_prior = 1

# 누적 성공 횟수와 실패 횟수 계산하여 추가
df["누적 성공 횟수"] = df["성공 여부"].cumsum()
df["누적 실패 횟수"] = (df["시도"] - df["누적 성공 횟수"])

# Directory to save images
output_dir = 'beta_distributions'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List to store file paths for GIF creation
filenames = []

# Step 4: Precompute the maximum y-axis value across all trials
p_values = np.linspace(0, 1, 100)
max_density = 0  # Variable to track the global maximum density

for _, row in df.iterrows():
    successes = row['누적 성공 횟수']
    failures = row['누적 실패 횟수']
    
    # Calculate posterior distribution parameters
    alpha_post, beta_post = calculate_posterior(successes, failures, alpha_prior, beta_prior)
    
    # Calculate the PDF of the Beta distribution
    beta_pdf_post = beta.pdf(p_values, alpha_post, beta_post)
    
    # Update the maximum density
    max_density = max(max_density, beta_pdf_post.max())

# Add labels and title for the prior distribution only
plt.figure(figsize=(5, 3))
beta_pdf_prior = beta.pdf(p_values, alpha_prior, beta_prior)
plt.plot(p_values, beta_pdf_prior, 
         label=f"사전 분포: Beta({alpha_prior}, {beta_prior})", 
         color="gray", lw=2, linestyle="--")
plt.xlim(0, 1)
plt.ylim(0, max_density * 1.1)  # Set y-axis slightly above the global max density
plt.title("각 시도에서의 사후 분포", fontsize=8)
plt.xlabel("성공 확률", fontsize=8)
plt.ylabel("성공 확률의 확률 밀도", fontsize=8)
plt.legend(loc="upper right", fontsize=6)
plt.tight_layout()

# Save the plot to a file (0th frame)
filename = f"{output_dir}/trial_0.png"
plt.savefig(filename, dpi=100)
filenames.append(filename)
plt.close()


# 매 시도에서 데이터를 반영하여 사후 분포 계산 및 그리기
for i, row in df.iterrows():
    plt.figure(figsize=(5, 3))
    beta_pdf_prior = beta.pdf(p_values, alpha_prior, beta_prior)
    plt.plot(p_values, beta_pdf_prior, 
             label=f"사전 분포: Beta({alpha_prior}, {beta_prior})", 
             color="gray", lw=2, linestyle="--")
    
    successes = row["누적 성공 횟수"]
    failures = row["누적 실패 횟수"]
    alpha_post, beta_post = calculate_posterior(successes, failures, alpha_prior, beta_prior)    
    beta_pdf_post = beta.pdf(p_values, alpha_post, beta_post)
    opacity = (i + 1) / len(df)    
    plt.plot(p_values, beta_pdf_post, 
             label=f"사후 분포 {int(row['시도'])}: Beta({alpha_post}, {beta_post})", 
             color="black", alpha=opacity, lw=1)
    
    final_alpha_post = alpha_post
    final_beta_post = beta_post
    expected_value = final_alpha_post / (final_alpha_post + final_beta_post)
    plt.axvline(expected_value, 
                label=f"최종 분포의 기댓값: {expected_value:.3f}",
                color="red", lw=2, linestyle='-.')
    plt.text(expected_value, -0.05, f'{expected_value:.2f}', ha='center', va='top', fontsize=8, color='red', transform=plt.gca().get_xaxis_transform())

    plt.xlim(0, 1)
    plt.ylim(0, max_density * 1.1)  # Set y-axis slightly above the global max density
    plt.title("각 시도에서의 사후 분포", fontsize=8)
    plt.xlabel("성공 확률", fontsize=8)
    plt.ylabel("성공 확률의 확률 밀도", fontsize=8)
    plt.legend(loc="upper right", fontsize=6)
    plt.tight_layout()
    filename = f"{output_dir}/trial_{i+1}.png"
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()

with imageio.get_writer('beta_distribution_updates.gif', mode='I', duration=1, loop=0) as writer:  # 1.0 sec delay
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    last_frame = imageio.imread(filenames[-1])

# for filename in filenames:
#     os.remove(filename)




