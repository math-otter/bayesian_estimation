import gacha
import matplotlib.pyplot as plt
import os
from PIL import Image

# 출력 옵션 설정
gacha.set_display_options(display_all_rows=True, display_all_columns=True)

# 가챠 설정, 결과 생성
seed = 42
p = 0.1  # 성공 확률
t = 100  # 시도 횟수
gacha_results = gacha.gacha_simulation(seed, p, t)  # 가챠 결과 생성

# 가챠 결과 분석
prior_success = 0  # 사전 성공 횟수
prior_failure = 0  # 사전 실패 횟수
df, p_obs, beta_pdfs = gacha.gacha_analysis(gacha_results, 
                                            alpha_prior=1 + prior_success, 
                                            beta_prior=1 + prior_failure,
                                            p_obs_num=100)

# 폰트 설정
plt.rc("font", family="Malgun Gothic")

# 딜레이 설정
delay_start = 5000  # 사전 분포 딜레이 (밀리초 단위)
delay_middle = 100  # 중간 프레임 딜레이 (밀리초 단위)
delay_end = 5000  # 마지막 프레임 딜레이 (밀리초 단위)
durations = [delay_start] + [delay_middle] * (len(df) - 2) + [delay_end]

# 이미지를 저장할 디렉토리가 없으면 생성
if not os.path.exists("Images"):
    os.makedirs("Images")

# 시각화 및 이미지 저장 (동적으로 패딩 자리수 결정)
for i in range(len(df)):
    plt.plot(p_obs, beta_pdfs[i], 
             label=df["beta_dist"][i], 
             color="black", lw=2, linestyle="-")
    
    beta_mean = df["beta_mean"][i]
    beta_var = df["beta_var"][i]
    plt.axvline(beta_mean, 
                label=f"기댓값: {beta_mean:.3f} (분산: {beta_var:.3f})",
                color="red", lw=2, linestyle='--')
    plt.text(beta_mean, -0.005, f'{beta_mean:.3f}', 
             ha="center", va="top", fontsize=8, color="red", transform=plt.gca().get_xaxis_transform())

    plt.xlim(0, 1)
    ylim = df["beta_max"].max() * 1.1
    plt.ylim(0, ylim)
    plt.legend(loc="upper right")
    plt.xlabel("성공 확률")
    plt.ylabel("성공 확률의 확률 밀도")
    plt.title("각 시도에서 성공 확률의 확률 분포")
    plt.tight_layout()

    # 파일 이름을 동적으로 패딩 (필요한 자리수로 설정)
    plt.savefig(f"Images/{i:0{len(str(len(df)))}d}.png")
    plt.close()

# 폴더에서 파일명을 자동으로 리스트로 저장
folder_path = "Images"
filenames = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith(".png")]

# 이미지를 PIL로 불러와서 GIF 생성 (딜레이 반영)
images = [Image.open(filename) for filename in filenames]
images[0].save("beta_distribution_updates.gif", save_all=True, append_images=images[1:], duration=durations, loop=0)

# 생성된 이미지를 폴더와 함께 지우기
for filename in filenames:
    os.remove(filename)

# 폴더 삭제
os.rmdir(folder_path)