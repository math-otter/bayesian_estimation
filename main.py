import gacha_simulation as gacha

# 가챠 설정
seed = 42
p = 0.5  # 성공 확률
t = 5    # 시도 횟수

# 가챠 결과 생성
gacha_results = gacha.gacha_simulation(seed, p, t)

# 데이터프레임 생성
df = gacha.create_gacha_dataframe(gacha_results, alpha_prior=1+10, beta_prior=1+5)

# 데이터프레임 출력
print(df)