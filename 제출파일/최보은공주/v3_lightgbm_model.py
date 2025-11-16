"""
v3: LightGBM + Feature Engineering
- 임계값: 0.35 (정답 2,400개 근처)
- 모델: LightGBM
- Feature: 15개+
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 v3: LightGBM + Feature Engineering")
print("=" * 80)

# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================
print("\n[1단계] 데이터 로드 중...")
train = pd.read_excel('/mnt/user-data/uploads/train.xlsx')
train['ym'] = pd.to_datetime(train['year'].astype(str) + '-' + train['month'].astype(str) + '-01')

# Pivot 테이블 생성
pivot = train.groupby(['item_id', 'ym'])['value'].sum().unstack(fill_value=0)
print(f"Pivot shape: {pivot.shape}")

# ============================================================================
# 2. 공행성 쌍 탐색 (임계값 0.35)
# ============================================================================
print("\n[2단계] 공행성 쌍 탐색 중 (임계값 0.35)...")

def find_comovement_pairs_v3(pivot, corr_threshold=0.35, max_lag=6):
    """
    공행성 쌍 탐색 - v3
    """
    items = pivot.index.tolist()
    pairs = []
    
    for i, leader in enumerate(tqdm(items, desc="공행성 탐색")):
        a_series = pivot.loc[leader].values
        
        for follower in items:
            if leader == follower:
                continue
            
            b_series = pivot.loc[follower].values
            
            best_corr = -1
            best_lag = 0
            
            for lag in range(max_lag + 1):
                if lag >= len(a_series):
                    break
                
                a_lagged = a_series[:-lag-1] if lag > 0 else a_series[:-1]
                b_target = b_series[lag+1:]
                
                if len(a_lagged) != len(b_target) or len(a_lagged) < 2:
                    continue
                
                # Pearson 상관계수
                corr = np.corrcoef(a_lagged, b_target)[0, 1]
                
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_lag = lag
            
            if best_corr >= corr_threshold:
                pairs.append({
                    'leading_item_id': leader,
                    'following_item_id': follower,
                    'max_corr': best_corr,
                    'best_lag': best_lag
                })
    
    return pd.DataFrame(pairs)

pairs = find_comovement_pairs_v3(pivot, corr_threshold=0.35, max_lag=6)
print(f"\n✅ 탐색된 공행성 쌍 수: {len(pairs)}")
print(f"상관계수 범위: {pairs['max_corr'].min():.3f} ~ {pairs['max_corr'].max():.3f}")

# ============================================================================
# 3. Feature Engineering
# ============================================================================
print("\n[3단계] Feature Engineering 중...")

def create_features_v3(pivot, pairs):
    """
    고급 Feature 생성
    """
    months = pivot.columns.to_list()
    n_months = len(months)
    train_data = []
    
    for row in tqdm(pairs.itertuples(index=False), total=len(pairs), desc="Feature 생성"):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)
        
        a_series = pivot.loc[leader].values.astype(float)
        b_series = pivot.loc[follower].values.astype(float)
        
        for t in range(lag + 6, n_months - 1):  # 충분한 과거 데이터 확보
            # 기본 features
            b_t = b_series[t]
            b_t_1 = b_series[t - 1]
            a_t_lag = a_series[t - lag]
            target = b_series[t + 1]
            
            # Lag features (다양한 lag)
            a_lag1 = a_series[t - 1] if t >= 1 else 0
            a_lag2 = a_series[t - 2] if t >= 2 else 0
            a_lag3 = a_series[t - 3] if t >= 3 else 0
            
            b_lag2 = b_series[t - 2] if t >= 2 else 0
            b_lag3 = b_series[t - 3] if t >= 3 else 0
            
            # 이동평균 (3개월, 6개월)
            b_ma3 = np.mean(b_series[max(0, t-2):t+1]) if t >= 2 else b_t
            b_ma6 = np.mean(b_series[max(0, t-5):t+1]) if t >= 5 else b_t
            
            a_ma3 = np.mean(a_series[max(0, t-2):t+1]) if t >= 2 else a_t_lag
            
            # 변화율 (MoM)
            b_mom = (b_t - b_t_1) / (b_t_1 + 1) if b_t_1 > 0 else 0
            a_mom = (a_t_lag - a_lag1) / (a_lag1 + 1) if a_lag1 > 0 else 0
            
            # 계절성 (월)
            month = pd.to_datetime(months[t]).month
            is_jan = 1 if month == 1 else 0
            is_sep = 1 if month == 9 else 0
            
            # 트렌드
            year = pd.to_datetime(months[t]).year
            year_effect = year - 2022
            
            train_data.append({
                # 기본 features
                'b_t': b_t,
                'b_t_1': b_t_1,
                'a_t_lag': a_t_lag,
                'max_corr': corr,
                'best_lag': float(lag),
                
                # Lag features
                'a_lag1': a_lag1,
                'a_lag2': a_lag2,
                'a_lag3': a_lag3,
                'b_lag2': b_lag2,
                'b_lag3': b_lag3,
                
                # 이동평균
                'b_ma3': b_ma3,
                'b_ma6': b_ma6,
                'a_ma3': a_ma3,
                
                # 변화율
                'b_mom': b_mom,
                'a_mom': a_mom,
                
                # 계절성
                'is_jan': is_jan,
                'is_sep': is_sep,
                'month': month,
                
                # 트렌드
                'year_effect': year_effect,
                
                'target': target
            })
    
    return pd.DataFrame(train_data)

df_train = create_features_v3(pivot, pairs)
print(f"\n학습 데이터 shape: {df_train.shape}")
print(f"Feature 개수: {len(df_train.columns) - 1}")

# ============================================================================
# 4. LightGBM 모델 학습
# ============================================================================
print("\n[4단계] LightGBM 모델 학습 중...")

feature_cols = [col for col in df_train.columns if col != 'target']
train_X = df_train[feature_cols].values
train_y = df_train['target'].values

# LightGBM 모델
model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

model.fit(train_X, train_y)
print("✅ 모델 학습 완료!")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 중요 Features:")
print(feature_importance.head(10))

# ============================================================================
# 5. 예측 및 제출 파일 생성
# ============================================================================
print("\n[5단계] 예측 중...")

def predict_v3(pivot, pairs, model, feature_cols):
    """
    v3 예측 함수
    """
    months = pivot.columns.to_list()
    n_months = len(months)
    t_last = n_months - 1
    t_prev = n_months - 2
    
    preds = []
    
    for row in tqdm(pairs.itertuples(index=False), total=len(pairs), desc="예측 중"):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)
        
        if leader not in pivot.index or follower not in pivot.index:
            continue
        
        a_series = pivot.loc[leader].values.astype(float)
        b_series = pivot.loc[follower].values.astype(float)
        
        if t_last - lag < 0:
            continue
        
        # 기본 features
        b_t = b_series[t_last]
        b_t_1 = b_series[t_prev]
        a_t_lag = a_series[t_last - lag]
        
        # Lag features
        a_lag1 = a_series[t_last - 1]
        a_lag2 = a_series[t_last - 2] if t_last >= 2 else 0
        a_lag3 = a_series[t_last - 3] if t_last >= 3 else 0
        b_lag2 = b_series[t_last - 2] if t_last >= 2 else 0
        b_lag3 = b_series[t_last - 3] if t_last >= 3 else 0
        
        # 이동평균
        b_ma3 = np.mean(b_series[max(0, t_last-2):t_last+1])
        b_ma6 = np.mean(b_series[max(0, t_last-5):t_last+1])
        a_ma3 = np.mean(a_series[max(0, t_last-2):t_last+1])
        
        # 변화율
        b_mom = (b_t - b_t_1) / (b_t_1 + 1) if b_t_1 > 0 else 0
        a_mom = (a_t_lag - a_lag1) / (a_lag1 + 1) if a_lag1 > 0 else 0
        
        # 계절성 (2025년 8월 예측)
        month = 8
        is_jan = 0
        is_sep = 0
        
        # 트렌드
        year_effect = 2025 - 2022
        
        # Feature 배열 생성
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
            'is_jan': is_jan,
            'is_sep': is_sep,
            'month': month,
            'year_effect': year_effect
        }
        
        X_test = np.array([[features[col] for col in feature_cols]])
        y_pred = model.predict(X_test)[0]
        
        # 후처리
        y_pred = max(0.0, float(y_pred))
        y_pred = int(round(y_pred))
        
        preds.append({
            'leading_item_id': leader,
            'following_item_id': follower,
            'value': y_pred
        })
    
    return pd.DataFrame(preds)

submission = predict_v3(pivot, pairs, model, feature_cols)
print(f"\n✅ 예측 완료!")
print(f"제출 파일 shape: {submission.shape}")
print(f"\n예측값 통계:\n{submission['value'].describe()}")

# 저장
output_path = '/mnt/user-data/outputs/v3_lightgbm_submission.csv'
submission.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n✅ 제출 파일 저장: {output_path}")

print("\n" + "=" * 80)
print("v3 생성 완료! 🎉")
print("=" * 80)
print(f"\nv2 vs v3 비교:")
print(f"  v2: 2,903개 쌍, Linear Regression, 5 features")
print(f"  v3: {len(submission)}개 쌍, LightGBM, {len(feature_cols)} features")
print(f"\n예상 점수:")
print(f"  v2: 0.25 ~ 0.30")
print(f"  v3: 0.35 ~ 0.45 (Feature + 모델 개선)")
print("=" * 80)
