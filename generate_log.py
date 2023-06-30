import os

import pandas as pd

from utils import mape, rmse
import warnings
warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
def generate_log(region):
    gt_df = pd.read_csv('./output/' + region + '_gt.csv', index_col=0)
    pred_df = pd.read_csv('./output/' + region + '_pred.csv', index_col=0)
    roi_df = pd.read_csv('./output/' + region + '_roi.csv', index_col=0)

    # print(gt_df)
    rmse_list = []
    mape_list = []
    roi_list = []
    id_list = []
    idx = []
    for i in range(len(gt_df.index)):
        accumulate_rmse = rmse(gt_df.iloc[:i].values, pred_df.iloc[:i].values)
        accumulate_mape = mape(gt_df.iloc[:i].values, pred_df.iloc[:i].values)
        print(i, '번 째 데이터 로그', '누적 RMSE : ', accumulate_rmse, ' 누적 MAPE : ', accumulate_mape)
        rmse_list.append(accumulate_rmse)
        mape_list.append(accumulate_mape)
        roi_list.append(roi_df.iloc[i].values)
        id_list.append(str(i))
        idx.append(region)
    data = {
        'Data ID': id_list,
        'RMSE (누적)': rmse_list,
        'MAPE (누적)': mape_list,
    }
    frame = pd.DataFrame(data, index=idx)
    frame.to_csv('./log/' + region + '모델 테스트 로그(Error).csv', encoding='utf-8')

    # roi_df['Data ID'] = roi_df.index
    # roi_df.insert(0, 'Data ID', roi_df.index)
    print(roi_df)
    roi_df.to_csv('./log/'+ region + '모델 테스트 로그(교통유발 시설물 카운트).csv', encoding='utf-8')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--region',type=str)
    args = parser.parse_args()
    generate_log(args.region)
