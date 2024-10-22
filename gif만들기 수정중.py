import numpy as np
import pandas as pd
from scipy.stats import beta

# Pandas 옵션을 설정하여 모든 열을 출력하고, 줄바꿈을 없애도록 충분한 출력 너비 설정
pd.set_option('display.max_columns', None)  # 모든 열을 출력
pd.set_option('display.width', 1000)  # 출력 너비를 넉넉하게 설정 (1000으로 설정)

# 1. Gacha 설정 함수
def gacha_simulation(seed, p, t):
    np.random.seed(seed)  # 랜덤 시드 고정
    return np.random.binomial(1, p, t)  # t개의 난수 생성 (성공: 1, 실패: 0)

# 2. 베타 분포에서 PDF의 최대값을 찾는 함수 (np.linspace(0, 1, 100) 사용)
def find_beta_max(alpha, beta_param):
    x = np.linspace(0, 1, 100)  # 0부터 1까지 100개의 값 생성
    pdf_values = beta.pdf(x, alpha, beta_param)  # 베타 분포의 확률밀도 함수 계산
    return np.max(pdf_values)  # 최대값 반환

# 3. 결과를 데이터프레임에 저장하는 함수
def create_gacha_dataframe(gacha_results, alpha_prior=1, beta_prior=1):
    # 사전 경험 데이터 (0행)
    cum_success = [alpha_prior - 1]
    cum_failure = [beta_prior - 1]
    cum_trials = [alpha_prior + beta_prior - 2]
    
    # 누적 성공 빈도 (0/0인 경우 NaN 처리)
    if cum_trials[0] == 0:
        cum_success_rate = [np.nan]
    else:
        cum_success_rate = [cum_success[0] / cum_trials[0]]
    
    # 베타 분포 정보 및 평균, 분산, 최대 확률밀도 추가
    beta_dist = ["Beta({}, {})".format(alpha_prior, beta_prior)]
    beta_mean = [alpha_prior / (alpha_prior + beta_prior)]
    beta_var = [(alpha_prior * beta_prior) / ((alpha_prior + beta_prior) ** 2 * (alpha_prior + beta_prior + 1))]
    beta_max = [find_beta_max(alpha_prior, beta_prior)]  # 베타 분포의 최대 확률밀도 값 추가
    
    # 가챠 결과 추가
    for i, result in enumerate(gacha_results):
        cum_success.append(cum_success[-1] + result)
        cum_failure.append(cum_failure[-1] + (1 - result))
        cum_trials.append(cum_trials[-1] + 1)
        cum_success_rate.append(cum_success[-1] / cum_trials[-1])
        
        # 각 시도에 따른 베타 분포 업데이트
        current_alpha = cum_success[-1] + 1  # alpha 값은 성공 횟수 + 1
        current_beta = cum_failure[-1] + 1   # beta 값은 실패 횟수 + 1
        beta_dist.append("Beta({}, {})".format(current_alpha, current_beta))
        
        # 베타 분포 평균, 분산, 최대 확률밀도 계산
        beta_mean.append(current_alpha / (current_alpha + current_beta))
        beta_var.append((current_alpha * current_beta) / ((current_alpha + current_beta) ** 2 * (current_alpha + current_beta + 1)))
        beta_max.append(find_beta_max(current_alpha, current_beta))
    
    # success_rate_diff 열을 계산 (빈도주의 성공 빈도 - 베타분포 기댓값)
    success_rate_diff = [cum_success_rate[i] - beta_mean[i] for i in range(len(cum_success_rate))]
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'trial': np.arange(len(gacha_results) + 1),
        'result': [np.nan] + gacha_results.tolist(),
        'cum_trials': cum_trials,
        'cum_success': cum_success,
        'cum_failure': cum_failure,
        'cum_success_rate': cum_success_rate,
        'success_rate_diff': success_rate_diff,  # cum_success_rate 바로 다음에 diff 추가
        'beta_dist': beta_dist,
        'beta_mean': beta_mean,
        'beta_var': beta_var,
        'beta_max': beta_max  # 확률밀도 함수 최대값 열 추가
    })
    
    return df

# 가챠 설정
seed = 42
p = 0.5  # 성공 확률
t = 5    # 시도 횟수

# 가챠 결과 생성
gacha_results = gacha_simulation(seed, p, t)

# 데이터프레임 생성
df = create_gacha_dataframe(gacha_results, alpha_prior=1+1, beta_prior=1+1)
print(df)
