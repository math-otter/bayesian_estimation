import numpy as np
import pandas as pd
from scipy.stats import beta

# 가챠 시뮬레이션 함수
def gacha_simulation(seed, p, t):
    """
    가챠 시뮬레이션을 통해 t번의 시행에서 성공(1) 또는 실패(0)의 결과를 반환합니다.
    """
    np.random.seed(seed)  # 랜덤 시드 고정
    return np.random.binomial(1, p, t)  # t개의 난수 생성 (성공: 1, 실패: 0)

# 베타 분포의 확률 밀도 함수의 최대값을 찾는 함수
def find_beta_max(alpha, beta_param):
    """
    주어진 알파와 베타 값에 대해 베타 분포의 확률 밀도 함수의 최대값을 찾습니다.
    """
    x = np.linspace(0, 1, 100)  # 0부터 1까지 100개의 값 생성
    pdf_values = beta.pdf(x, alpha, beta_param)  # 베타 분포의 확률밀도 함수 계산
    return np.max(pdf_values)  # 최대값 반환

# 결과를 데이터프레임에 저장하는 함수
def create_gacha_dataframe(gacha_results, alpha_prior=1, beta_prior=1):
    """
    가챠 결과를 기반으로 누적 성공/실패 횟수와 베타 분포 통계량을 계산하여 데이터프레임으로 반환합니다.
    """
    cum_success = [alpha_prior - 1]
    cum_failure = [beta_prior - 1]
    cum_trials = [alpha_prior + beta_prior - 2]
    
    if cum_trials[0] == 0:
        cum_success_rate = [np.nan]
    else:
        cum_success_rate = [cum_success[0] / cum_trials[0]]
    
    beta_dist = ["Beta({}, {})".format(alpha_prior, beta_prior)]
    beta_mean = [alpha_prior / (alpha_prior + beta_prior)]
    beta_var = [(alpha_prior * beta_prior) / ((alpha_prior + beta_prior) ** 2 * (alpha_prior + beta_prior + 1))]
    beta_max = [find_beta_max(alpha_prior, beta_prior)]
    
    for i, result in enumerate(gacha_results):
        cum_success.append(cum_success[-1] + result)
        cum_failure.append(cum_failure[-1] + (1 - result))
        cum_trials.append(cum_trials[-1] + 1)
        cum_success_rate.append(cum_success[-1] / cum_trials[-1])
        
        current_alpha = cum_success[-1] + 1
        current_beta = cum_failure[-1] + 1
        beta_dist.append("Beta({}, {})".format(current_alpha, current_beta))
        
        beta_mean.append(current_alpha / (current_alpha + current_beta))
        beta_var.append((current_alpha * current_beta) / ((current_alpha + current_beta) ** 2 * (current_alpha + current_beta + 1)))
        beta_max.append(find_beta_max(current_alpha, current_beta))
    
    success_rate_diff = [cum_success_rate[i] - beta_mean[i] for i in range(len(cum_success_rate))]
    
    df = pd.DataFrame({
        'trial': np.arange(len(gacha_results) + 1),
        'result': [np.nan] + gacha_results.tolist(),
        'cum_trials': cum_trials,
        'cum_success': cum_success,
        'cum_failure': cum_failure,
        'cum_success_rate': cum_success_rate,
        'success_rate_diff': success_rate_diff,
        'beta_dist': beta_dist,
        'beta_mean': beta_mean,
        'beta_var': beta_var,
        'beta_max': beta_max
    })
    
    return df