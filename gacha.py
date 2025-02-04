import numpy as np
import pandas as pd
from scipy.stats import beta

# Pandas 옵션을 설정하여 모든 행과 열을 출력할지 여부와 너비를 제어하는 함수
def set_display_options(display_all_rows=False, display_all_columns=False):
    if display_all_rows:
        pd.set_option("display.max_rows", None)
    else:
        pd.reset_option("display.max_rows")
    
    if display_all_columns:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)  # 모든 열을 출력할 때 출력 너비를 넉넉하게 설정
    else:
        pd.reset_option("display.max_columns")
        pd.reset_option("display.width")

# 가챠 시뮬레이션 함수
def gacha_simulation(seed, p, t):
    np.random.seed(seed)
    gacha_results = np.random.binomial(1, p, t)
    return gacha_results

# 시뮬레이션 결과를 분석하여 반환하는 함수: 데이터프레임, 확률밀도함수값
def gacha_analysis(gacha_results, alpha_prior=1, beta_prior=1, p_obs_num=100):
    cum_success = [alpha_prior - 1]
    cum_failure = [beta_prior - 1]
    cum_trials = [alpha_prior + beta_prior - 2]
    
    if cum_trials[0] == 0:
        cum_success_rate = [np.nan]
    else:
        cum_success_rate = [cum_success[0] / cum_trials[0]]
    
    p_obs = np.linspace(0, 1, p_obs_num)
    beta_dist = ["Beta({}, {})".format(alpha_prior, beta_prior)]
    beta_pdf = [beta.pdf(p_obs, alpha_prior, beta_prior)]
    beta_mean = [alpha_prior / (alpha_prior + beta_prior)]
    beta_var = [(alpha_prior * beta_prior) / ((alpha_prior + beta_prior) ** 2 * (alpha_prior + beta_prior + 1))]
    beta_max = [np.max(beta.pdf(p_obs, alpha_prior, beta_prior))]
    
    for _, result in enumerate(gacha_results):
        cum_success.append(cum_success[-1] + result)
        cum_failure.append(cum_failure[-1] + (1 - result))
        cum_trials.append(cum_trials[-1] + 1)
        cum_success_rate.append(cum_success[-1] / cum_trials[-1])
        
        current_alpha = cum_success[-1] + 1
        current_beta = cum_failure[-1] + 1
        beta_dist.append("Beta({}, {})".format(current_alpha, current_beta))
        beta_pdf.append(beta.pdf(p_obs, current_alpha, current_beta))
        beta_mean.append(current_alpha / (current_alpha + current_beta))
        beta_var.append((current_alpha * current_beta) / ((current_alpha + current_beta) ** 2 * (current_alpha + current_beta + 1)))
        beta_max.append(np.max(beta.pdf(p_obs, current_alpha, current_beta)))
    
    # diff 열을 계산 (빈도주의 성공 빈도 - 베타분포 기댓값)
    diff = [cum_success_rate[i] - beta_mean[i] for i in range(len(cum_success_rate))]
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        "trial": np.arange(len(gacha_results) + 1),
        "result": [np.nan] + gacha_results.tolist(),
        "cum_trials": cum_trials,
        "cum_success": cum_success,
        "cum_failure": cum_failure,
        "cum_success_rate": cum_success_rate,
        "diff": diff, 
        "beta_dist": beta_dist,
        "beta_mean": beta_mean,
        "beta_var": beta_var,
        "beta_max": beta_max
    })

    # result 값을 0, 1로 정수화
    df["result"] = df["result"].fillna(np.nan).astype("Int64")
    
    return df, p_obs, beta_pdf