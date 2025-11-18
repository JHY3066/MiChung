"""
v8: v7 로직 + v3 Lagged Pearson 단일 필터 적용
- v7 (글로벌 RF 모델, v3 특성 공학)을 기반
- 공행성 필터: Lagged Pearson 단일 필터만 사용
- 임계값: Lagged Pearson > 0.30
- 모델: 단일 글로벌 RandomForest
- [Fix]: .itertuples() -> .iterrows()로 안정성 확보
- [수정]: 제출 파일을 9900개가 아닌, 필터된 쌍으로만 생성
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 v8: v7 로직 + Lagged Pearson 단일 필터 적용 (필터된 쌍만 제출)")
print("=" * 80)

# ============================================================================
# 1. 데이터 로드 및 피벗 테이블 생성
# ============================================================================
print("\n[1단계] 데이터 로드 중...")
try:
    train = pd.read_csv("train.csv")
    submission_df = pd.read_csv("sample_submission.csv")
except FileNotFoundError:
    print("❌ 오류: train.csv 또는 sample_submission.csv 파일을 찾을 수 없습니다.")
    raise SystemExit

# year, month → ym (월 단위 날짜)
train['ym'] = pd.to_datetime(
    train['year'].astype(str) + '-' + train['month'].astype(str) + '-01'
)

# item_id × ym 피벗테이블 (월별 value 합)
pivot = train.groupby(['item_id', 'ym'])['value'].sum().unstack(fill_value=0)

print(f"✅ Pivot shape: {pivot.shape}")

# ============================================================================
# 2. 공행성 쌍 탐색 (Lagged Pearson 단일 필터)
# ============================================================================
print("\n[2단계] 공행성 쌍 탐색 중 (Lagged Pearson 단일 필터, threshold 0.30)...")

def find_best_lag_corr_pearson(a_series_val, b_series_val, max_lag=12):
    """
    a_t (leader) 가 선행, b_{t+lag} (follower)를 예측한다고 보고,
    lag ∈ [0, max_lag] 범위에서 가장 큰 Pearson 상관계수와 해당 lag를 찾음.
    """
    best_corr = -1.0
    best_lag = 0
    n = len(a_series_val)

    for lag in range(max_lag + 1):  # lag: 0 ~ max_lag
        if lag >= n:
            break

        # a_t (선행), b_{t+lag} (후행)
        a_lagged = a_series_val[:-lag-1] if lag > 0 else a_series_val[:-1]
        b_target = b_series_val[lag+1:]

        if len(a_lagged) != len(b_target) or len(a_lagged) < 2:
            continue

        corr = np.corrcoef(a_lagged, b_target)[0, 1]
        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag

    return best_corr, best_lag

pairs = []
for index, row in tqdm(submission_df.iterrows(),
                       total=len(submission_df),
                       desc="쌍 관계 탐색 (Lagged Pearson)"):
    leader = row['leading_item_id']
    follower = row['following_item_id']

    if leader in pivot.index and follower in pivot.index:
        a_orig = pivot.loc[leader].values.astype(float)
        b_orig = pivot.loc[follower].values.astype(float)

        pearson_corr, best_lag = find_best_lag_corr_pearson(a_orig, b_orig, max_lag=12)

        pairs.append({
            'leading_item_id': leader,
            'following_item_id': follower,
            'max_corr': pearson_corr,  # Lagged Pearson
            'best_lag': best_lag
        })

pairs_df = pd.DataFrame(pairs)

CORR_THRESHOLD_PEARSON = 0.35

filtered_pairs = pairs_df[
    (pairs_df['max_corr'] >= CORR_THRESHOLD_PEARSON)
].copy()

print(f"\n✅ 탐색된 공행성 쌍 수 (Lagged Pearson 필터 통과): {len(filtered_pairs)}")
if len(filtered_pairs) > 0:
    print(filtered_pairs['max_corr'].describe())
else:
    print("⚠️ 필터 통과 쌍 없음 (threshold를 낮춰볼 필요가 있을 수 있음)")

# ============================================================================
# 3. Feature Engineering (v3 스타일)
# ============================================================================
print("\n[3단계] Feature Engineering 중...")

def create_features_v3(pivot_table, pairs_to_train):
    """
    v3와 동일한 Feature 구성 (Lagged Pearson만 사용)
    """
    months = pivot_table.columns.to_list()
    n_months = len(months)
    train_data = []

    for index, row in tqdm(pairs_to_train.iterrows(),
                           total=len(pairs_to_train),
                           desc="Feature 생성"):
        leader = row['leading_item_id']
        follower = row['following_item_id']
        lag = int(row['best_lag'])
        corr = float(row['max_corr'])

        a_series = pivot_table.loc[leader].values.astype(float)
        b_series = pivot_table.loc[follower].values.astype(float)

        # t 시점에서 t+1을 예측하는 학습셋 생성
        for t in range(lag + 6, n_months - 1):  # 최소 lag + 6개월 확보
            b_t = b_series[t]
            b_t_1 = b_series[t - 1]
            a_t_lag = a_series[t - lag]

            # target: log1p 변환
            target = np.log1p(b_series[t + 1])

            # Lag features
            a_lag1 = a_series[t - 1]
            a_lag2 = a_series[t - 2] if t >= 2 else 0.0
            a_lag3 = a_series[t - 3] if t >= 3 else 0.0

            b_lag2 = b_series[t - 2] if t >= 2 else 0.0
            b_lag3 = b_series[t - 3] if t >= 3 else 0.0

            # 이동평균
            b_ma3 = np.mean(b_series[max(0, t-2):t+1])
            b_ma6 = np.mean(b_series[max(0, t-5):t+1])
            a_ma3 = np.mean(a_series[max(0, t-2):t+1])

            # MoM 변화율
            b_mom = (b_t - b_t_1) / (b_t_1 + 1.0) if b_t_1 > 0 else 0.0
            a_mom = (a_t_lag - a_lag1) / (a_lag1 + 1.0) if a_lag1 > 0 else 0.0

            # 달/연 효과
            month_dt = pd.to_datetime(months[t])
            month = month_dt.month
            year = month_dt.year
            year_effect = year - 2022

            train_data.append({
                'b_t': b_t,
                'b_t_1': b_t_1,
                'a_t_lag': a_t_lag,
                'max_corr': corr,
                'best_lag': float(lag),
                'a_lag1': a_lag1,
                'a_lag2': a_lag2,
                'a_lag3': a_lag3,
                'b_lag2': b_lag2,
                'b_lag3': b_lag3,
                'b_ma3': b_ma3,
                'b_ma6': b_ma6,
                'a_ma3': a_ma3,
                'b_mom': b_mom,
                'a_mom': a_mom,
                'month': month,
                'year_effect': year_effect,
                'target': target
            })

    return pd.DataFrame(train_data)

if len(filtered_pairs) > 0:
    df_train = create_features_v3(pivot, filtered_pairs)
    feature_cols = [c for c in df_train.columns if c != 'target']
else:
    df_train = pd.DataFrame()
    feature_cols = []

print(f"✅ 학습 데이터 shape: {df_train.shape}")
print(f"✅ Feature 개수: {len(feature_cols)}")

# ============================================================================
# 4. RandomForest 모델 학습
# ============================================================================
print("\n[4단계] RandomForest 모델 학습 중...")

if not df_train.empty:
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(df_train[feature_cols].values, df_train['target'].values)
    print("✅ 모델 학습 완료!")

    # Feature Importance 간단 출력
    fi = (
        pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        })
        .sort_values('importance', ascending=False)
    )
    print("\n📊 Top 10 Feature Importance:")
    print(fi.head(10).to_string(index=False))
else:
    model = None
    print("⚠️ 학습 데이터가 없어 모델 학습을 건너뜁니다.")

# ============================================================================
# 5. 예측 (Lagged Pearson 필터 통과 쌍에만 RF 적용)
# ============================================================================
print("\n[5단계] 예측 중...")

def predict_v3(pivot_table, pairs_to_predict, model_to_use, feature_cols_list):
    """
    마지막 관측 월(t_last) 기준으로 t_last+1을 예측하는 v3 스타일 inference
    """
    months = pivot_table.columns.to_list()
    n_months = len(months)
    t_last = n_months - 1
    t_prev = n_months - 2

    preds = {}

    for index, row in tqdm(pairs_to_predict.iterrows(),
                           total=len(pairs_to_predict),
                           desc="RF 예측"):
        leader = row['leading_item_id']
        follower = row['following_item_id']
        lag = int(row['best_lag'])
        corr = float(row['max_corr'])

        if leader not in pivot_table.index or follower not in pivot_table.index:
            continue
        if t_last - lag < 0:
            continue

        a_series = pivot_table.loc[leader].values.astype(float)
        b_series = pivot_table.loc[follower].values.astype(float)

        # 기본 값
        b_t = b_series[t_last]
        b_t_1 = b_series[t_prev]
        a_t_lag = a_series[t_last - lag]

        a_lag1 = a_series[t_last - 1]
        a_lag2 = a_series[t_last - 2] if t_last >= 2 else 0.0
        a_lag3 = a_series[t_last - 3] if t_last >= 3 else 0.0

        b_lag2 = b_series[t_last - 2] if t_last >= 2 else 0.0
        b_lag3 = b_series[t_last - 3] if t_last >= 3 else 0.0

        b_ma3 = np.mean(b_series[max(0, t_last-2):t_last+1])
        b_ma6 = np.mean(b_series[max(0, t_last-5):t_last+1])
        a_ma3 = np.mean(a_series[max(0, t_last-2):t_last+1])

        b_mom = (b_t - b_t_1) / (b_t_1 + 1.0) if b_t_1 > 0 else 0.0
        a_mom = (a_t_lag - a_lag1) / (a_lag1 + 1.0) if a_lag1 > 0 else 0.0

        # 예측 대상 월: 2025년 8월 가정 (문제 설정에 맞게 고정)
        month = 8
        year_effect = 2025 - 2022

        features = {
            'b_t': b_t,
            'b_t_1': b_t_1,
            'a_t_lag': a_t_lag,
            'max_corr': corr,
            'best_lag': float(lag),
            'a_lag1': a_lag1,
            'a_lag2': a_lag2,
            'a_lag3': a_lag3,
            'b_lag2': b_lag2,
            'b_lag3': b_lag3,
            'b_ma3': b_ma3,
            'b_ma6': b_ma6,
            'a_ma3': a_ma3,
            'b_mom': b_mom,
            'a_mom': a_mom,
            'month': month,
            'year_effect': year_effect
        }

        X_test = np.array([[features[col] for col in feature_cols_list]])

        # 모델은 log1p 스케일에서 예측 → expm1로 복원
        y_log = model_to_use.predict(X_test)[0]
        y_pred = np.expm1(y_log)

        # 음수 방지 + 반올림
        y_pred = max(0.0, float(y_pred))
        y_pred = int(round(y_pred))

        preds[(leader, follower)] = y_pred

    return preds

if model is not None and not filtered_pairs.empty and len(feature_cols) > 0:
    predictions_dict = predict_v3(pivot, filtered_pairs, model, feature_cols)
    print(f"\n✅ RF 예측 완료! {len(predictions_dict)}개의 (leader, follower) 쌍에 대해 예측했습니다.")
else:
    predictions_dict = {}
    print("⚠️ 모델 또는 필터된 쌍이 없어 RF 예측을 건너뜁니다.")

# ============================================================================
# 6. 제출 파일 생성 (필터된 쌍만)
# ============================================================================
print("\n[6단계] 제출 파일 생성 중 (필터된 쌍만)...")

# 1. filtered_pairs DataFrame에 예측값('value')을 매핑
filtered_pairs_with_preds = filtered_pairs.copy()
filtered_pairs_with_preds['value'] = filtered_pairs_with_preds.apply(
    lambda row: predictions_dict.get(
        (row['leading_item_id'], row['following_item_id']), 0
    ),
    axis=1
)

# 2. 요청하신 '제출용' 파일 저장 (필터된 쌍만)
output_path = "submission_single_filter_pearson_filtered_only_0.35.csv"
final_submission_df = filtered_pairs_with_preds[[
    'leading_item_id', 'following_item_id', 'value'
]]

final_submission_df.to_csv(output_path, index=False)

print("\n🎉 완료!")
print(f"✅ [제출용] 필터된 쌍 파일 저장: {output_path}")
print("=" * 80)
print(f"총 {len(final_submission_df)}개의 쌍을 예측하여 파일에 저장했습니다.")
print(f"예측값 통계 (0 포함):")
if len(predictions_dict) > 0:
    print(final_submission_df['value'].describe())
else:
    print("0보다 큰 예측값이 없습니다.")
print("=" * 80)
