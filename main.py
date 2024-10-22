import gacha_simulation as gacha

# 출력 옵션 설정
gacha.set_display_options(display_all_rows=True, display_all_columns=True)

# 가챠 설정
seed = 42
p = 0.5  # 성공 확률
t = 50    # 시도 횟수

# 가챠 결과 생성
gacha_results = gacha.gacha_simulation(seed, p, t)

# 데이터프레임 생성
df = gacha.create_gacha_dataframe(gacha_results, alpha_prior=1+0, beta_prior=1+0)

# 데이터프레임 출력, 저장
print(df)
df.to_csv("result.csv", index=False)