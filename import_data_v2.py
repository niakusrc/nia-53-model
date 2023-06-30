import copy
import pickle
import pandas as pd
import cv2
import numpy as np
import datetime
import warnings
warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
# np.set_printoptions(threshold=np.inf)
def load_test_data(region):
    print('#### Data Loading Process Start ####')
    print('Time is : ', datetime.datetime.now())
    demand_data = pd.read_pickle("./preprocessing_v2/test/"+region+"_DemandData_1step.pkl")
    # demand_data = pd.read_pickle("data/total/DemandData_normal.pkl") ## last used
    # demand_data = pd.read_pickle("data/total/DemandData_1step.pkl")
    demand_data = demand_data.T
    print(region+'지역 test demand data type : ',type(demand_data))
    print(region+'지역 test demand data length : ', len(demand_data))
    print(region+'지역 test demand data shape : ', demand_data.shape)

    print(region+'지역 마스크 필터를 demand 데이터에 적용합니다.')
    mask = cv2.imread('./region/'+region+'filter.jpg',cv2.IMREAD_GRAYSCALE)
    # print('mask : ',mask)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] > 127:
                mask[i][j] = 1
            else:
                mask[i][j] = 0
    # print(mask)
    for idx,eliment in enumerate(demand_data):
        demand_data[idx] = np.multiply(eliment,mask)
    print(region + '지역 마스크 필터를 demand 데이터에 적용하였습니다.')
    # print(demand_data[0])
    demand_x = []
    demand_y = []
    tmp = []
    print(region + '지역 demand 데이터의 입력 데이터와 정답 데이터를 분할합니다.')
    for idx,data in enumerate(demand_data):
        # print(idx,'번 째 demand 데이터를 ',len(demand_x),'번째 시계열 묶음에 추가합니다.')
        tmp.append(data)
        if len(tmp) == 4:
            demand_x.append(tmp)
            # print(len(demand_x), '번째 시계열 묶음을 입력 데이터 리스트에 추가합니다.')
            if idx != len(demand_data) - 1:
                demand_y.append(demand_data[idx + 1])
                # print(idx+1,'번 째 demand 데이터를 ',len(demand_y),'번째 정답 데이터 리스트에 추가합니다.')
            else:
                demand_y.append(demand_data[idx])
                # print('마지막 demand 데이터를 ', len(demand_y), '번째 정답 데이터 리스트에 추가합니다.')
            tmp = []
    print(region + '지역 demand 데이터의 입력 데이터와 정답 데이터를 분할하였습니다.')
    print('demand_y shape : ',np.array(demand_y).shape)

    demand_x = np.array(demand_x).swapaxes(1, 3)
    supply_x = []
    for i in range(len(demand_x)):
        tmp = []
        for j in range(8):
            if 8+(i*4)+j == len(demand_x):
                tmp.append(demand_data[8 + (i * 4) + j-1])
            else:
                tmp.append(demand_data[-1])
        supply_x.append(tmp)
        tmp = []
    supply_x = np.array(supply_x)
    guide_x = []
    guide_data = pd.read_pickle('./preprocessing_v2/test/'+region+'_GuideData_1step.pkl')
    guide_data = guide_data.T
    guide_x = guide_data[::4]
    guide_x = guide_x.reshape(len(guide_x),57)
    roi_data = pd.read_pickle('./preprocessing_v2/test/'+region+'_ROIData_1step.pkl')
    roi_input = copy.deepcopy(roi_data)
    roi_input = roi_input.T
    roi_input.reshape(len(roi_input), 57)
    roi_input1 = roi_input[0::4]
    roi_input2 = roi_input[1::4]
    roi_input3 = roi_input[2::4]
    roi_input4 = roi_input[3::4]
    roi_x = [x + y + z + w for x, y, z, w in zip(roi_input1, roi_input2, roi_input3, roi_input4)]
    roi_x = np.array(roi_x)
    roi_x = roi_x.reshape(len(roi_x), 57)
    x_test = []
    y_test = []
    demand_y = np.array(demand_y)
    x_test.append(demand_x[:])
    x_test.append(supply_x[:])
    x_test.append(guide_x[:])
    x_test.append(roi_x[:])
    y_test = demand_y[:, :, :].reshape(int(len(demand_y[:, :, :])), 8, 8, 1)
    print('split_target time : ',int(len(demand_y)))

    print('#### Data Loading Process Start ####')
    print('Time is : ', datetime.datetime.now())
    return  x_test, y_test

def load_data(region):
    print('#### Data Loading Process Start ####')
    print('Time is : ', datetime.datetime.now())
    demand_data = pd.read_pickle("./preprocessing_v2/total/"+region+"_DemandData_1step.pkl")
    # demand_data = pd.read_pickle("data/total/DemandData_normal.pkl") ## last used
    # demand_data = pd.read_pickle("data/total/DemandData_1step.pkl")
    demand_data = demand_data.T
    print('type of data',type(demand_data))
    print('length of data :', len(demand_data))

    mask = cv2.imread('./region/'+region+'filter.jpg',cv2.IMREAD_GRAYSCALE)
    # print('mask : ',mask)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] > 127:
                mask[i][j] = 1
            else:
                mask[i][j] = 0
    # print(mask)
    for idx,eliment in enumerate(demand_data):
        demand_data[idx] = np.multiply(eliment,mask)
    # print(demand_data[0])
    demand_x = []
    demand_y = []
    tmp = []
    for idx,data in enumerate(demand_data):
        tmp.append(data)
        if len(tmp) == 4:
            demand_x.append(tmp)
            if idx != len(demand_data) - 1:
                # print(idx)
                demand_y.append(demand_data[idx + 1])
            else:
                demand_y.append(demand_data[idx])
            tmp = []

    # print(‘dmand_x_transpose’,np.array(demand_x))
    # print('demand_x:')
    demand_x = np.array(demand_x).swapaxes(1, 3)
    # print(demand_x.shape)
    # print('demand_y:')
    # print(np.array(demand_y).shape)
    supply_x = []
    # print('demand_x:',len(demand_x))
    # print('demand_data shape : ',demand_data.shape)
    for i in range(len(demand_x)):
        tmp = []
        # print('start at %s' %i)
        # if 8 + (i * 4) +8 > len(demand_x)-1:
        #     break
        for j in range(8):

            # print(8+(i*4)+j)
            if 8+(i*4)+j == len(demand_x):
                tmp.append(demand_data[8 + (i * 4) + j-1])
            else:
                tmp.append(demand_data[-1])
        # print('end')
        supply_x.append(tmp)
        tmp = []
        # supply_x.append(demand_data[8+(i * 4):(i * 4)])
        # print(‘appended : ’,demand_data[i*8:(i*8)+8])
        # print(‘appended : ’, demand_data[i * 8:(i * 8) + 8].shape)
    # print('supply_x:')
    supply_x = np.array(supply_x)
    # print(supply_x.shape)
    # supply_x = demand_x
    guide_x = []
    guide_data = pd.read_pickle('./preprocessing_v2/total/'+region+'_GuideData_1step.pkl')
    # guide_data = pd.read_pickle('data/total/GuideData_normal.pkl')
    guide_data = guide_data.T
    # print(guide_data)
    # print(guide_data[0])
    guide_x = guide_data[::4]
    guide_x = guide_x.reshape(len(guide_x),57)
    ## roi data ##
    roi_data = pd.read_pickle('./preprocessing_v2/total/'+region+'_ROIData_1step.pkl')
    # guide_data = pd.read_pickle('data/total/GuideData_normal.pkl')
    roi_input = copy.deepcopy(roi_data)
    roi_input = roi_input.T
    roi_input.reshape(len(roi_input), 57)
    roi_input1 = roi_input[0::4]
    roi_input2 = roi_input[1::4]
    roi_input3 = roi_input[2::4]
    roi_input4 = roi_input[3::4]

    roi_x = [x + y + z + w for x, y, z, w in zip(roi_input1, roi_input2, roi_input3, roi_input4)]

    roi_x = np.array(roi_x)
    roi_x = roi_x.reshape(len(roi_x), 57)
    ## roi data ##
    # print('guide_x:')
    # print(np.array(guide_x).shape)
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_test = []
    y_test = []
    # x_train.append(np.concatenate((demand_x[:int(len(demand_x)*0.5)-1],demand_x[int(len(demand_x)*0.75):]),axis=0))
    # x_train.append(np.concatenate((supply_x[:int(len(supply_x)*0.5)-1],supply_x[int(len(supply_x)*0.75):]),axis=0))
    # x_train.append(np.concatenate((guide_x[:int(len(guide_x)*0.5)-1],guide_x[int(len(guide_x)*0.75):]),axis=0))
    # print(len(demand_x)*0.75)
    x_train.append(demand_x[:int(len(demand_x)*0.8)])
    x_train.append(supply_x[:int(len(supply_x)*0.8)])
    x_train.append(guide_x[:int(len(guide_x)*0.8)])
    x_train.append(roi_x[:int(len(roi_x)*0.8)])
    demand_y = np.array(demand_y)
    # print('int(len(demand_y)*0.75) : ',int(len(demand_y)*0.75))
    y_train = (demand_y[:int(len(demand_y)*0.8), :, :]).reshape(int(len(demand_y)*0.8), 8, 8, 1)
    # print(int(len(demand_y)*0.75))
    # print('int(len(demand_y)*0.5)',int(len(demand_y)*0.5))
    # print('demand_y[:int(len(demand_y)*0.5), :, :]',len(demand_y[:int(len(demand_y)*0.5), :, :]))
    # print('demand_y[int(len(demand_y)*0.75):, :, :]',len(demand_y[int(len(demand_y)*0.75):, :, :]))
    # print('int(len(demand_y)*0.75)',int(len(demand_y)*0.75))
    # y_train = np.concatenate((demand_y[:int(len(demand_y)*0.5)-1, :, :],demand_y[int(len(demand_y)*0.75):, :, :]),axis=0).reshape(int(len(demand_y)*0.75), 8, 8, 1)
    # y_train = demand_y[0:9374]
    x_valid.append(demand_x[int(len(demand_x)*0.8):int(len(demand_x)*0.9)])
    x_valid.append(supply_x[int(len(supply_x)*0.8):int(len(supply_x)*0.9)])
    x_valid.append(guide_x[int(len(guide_x)*0.8):int(len(guide_x)*0.9)])
    x_valid.append(roi_x[int(len(roi_x)*0.8):int(len(roi_x)*0.9)])
    # y_valid = demand_y[9374:]
    # print('demand_y[int(len(demand_y)*0.5):int(len(demand_y)*0.75), :, :]',len(demand_y[int(len(demand_y)*0.5)-1:int(len(demand_y)*0.75), :, :]))
    # y_valid = demand_y[int(len(demand_y)*0.8):int(len(demand_y) * 0.9), :, :].reshape(len(demand_y) - int(len(demand_y) * 0.9), 8, 8, 1)
    y_valid = demand_y[int(len(demand_y) * 0.8):int(len(demand_y) * 0.9), :, :].reshape(int(len(demand_y[int(len(demand_y) * 0.8):int(len(demand_y) * 0.9), :, :])), 8, 8, 1)
    # print(int(len(demand_y)),int(len(demand_y[int(len(demand_y) * 0.8):int(len(demand_y) * 0.9), :, :])))
    # print(x_train[0].shape,x_train[1].shape,x_train[2].shape, y_train.shape, x_valid[0].shape,x_valid[1].shape,x_valid[2].shape, y_valid.shape)
    x_test.append(demand_x[int(len(demand_x)*0.9):])
    x_test.append(supply_x[int(len(supply_x)*0.9):])
    x_test.append(guide_x[int(len(guide_x)*0.9):])
    x_test.append(roi_x[int(len(roi_x)*0.9):])
    # y_test = demand_y[int(len(demand_y) * 0.9):, :, :].reshape(len(demand_y) - int(len(demand_y) * 0.9), 8, 8, 1)
    # print(int(len(demand_y)), int(len(demand_y[int(len(demand_y) * 0.9):, :, :])))
    y_test = demand_y[int(len(demand_y) * 0.9):, :, :].reshape(int(len(demand_y[int(len(demand_y) * 0.9):, :, :])), 8, 8, 1)
    print('split_target time : ',int(len(demand_y) * 0.9))

    print('#### Data Loading Process Start ####')
    print('Time is : ', datetime.datetime.now())
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_prediction_data(region):
    print('#### Data Loading Process Start ####')
    print('Time is : ', datetime.datetime.now())
    # demand_data = pd.read_pickle("preprocessing/"+region+"_DemandData_1step.pkl")
    # # demand_data = pd.read_pickle("data/prediction/DemandData_normal.pkl")
    # demand_data = demand_data.T
    # print('type of data',type(demand_data))
    #
    # mask = cv2.imread('region/'+region+'filter.jpg',cv2.IMREAD_GRAYSCALE)
    # for i in range(len(mask)):
    #     for j in range(len(mask[0])):
    #         if mask[i][j] > 127:
    #             mask[i][j] = 1
    #         else:
    #             mask[i][j] = 0
    # # print(mask)
    # for idx,eliment in enumerate(demand_data):
    #     demand_data[idx] = np.multiply(eliment,mask)
    # # print(demand_data[0])
    # demand_x = []
    # demand_y = []
    # tmp = []
    # for idx,data in enumerate(demand_data):
    #     if idx <8:
    #         continue
    #     else:
    #         tmp.append(data)
    #         if len(tmp) == 4:
    #             demand_x.append(tmp)
    #             if idx != len(demand_data)-1:
    #                 # print(idx)
    #                 demand_y.append(demand_data[idx+1])
    #             else:
    #                 demand_y.append(demand_data[idx])
    #             tmp = []
    #
    # # print(‘dmand_x_transpose’,np.array(demand_x))
    # print('demand_x:')
    # demand_x = np.array(demand_x).swapaxes(1, 3)
    # print(demand_x.shape)
    # print('demand_y:')
    # print(np.array(demand_y).shape)
    # supply_x = []
    # for i in range(int((len(demand_data)-8)/4)):
    #     supply_x.append(demand_data[i * 4:(i * 4) + 8])
    #     # print(‘appended : ’,demand_data[i*8:(i*8)+8])
    #     # print(‘appended : ’, demand_data[i * 8:(i * 8) + 8].shape)
    # print('supply_x:')
    # supply_x = np.array(supply_x)
    # guide_x = []
    # guide_data = pd.read_pickle('preprocessing/'+region+'_GuideData_1step.pkl')
    # # guide_data = pd.read_pickle('data/prediction/GuideData_normal.pkl')
    # guide_data = guide_data.T
    # # print(guide_data)
    # # print(guide_data[0])
    # guide_x = guide_data[8::4]
    # guide_x = guide_x.reshape(len(guide_x),57)
    # print('guide_x:')
    # print(np.array(guide_x).shape)
    # x_train = []
    # y_train = []
    # x_valid = []
    # y_valid = []
    # x_train.append(demand_x[:])
    # x_train.append(supply_x[:])
    # x_train.append(guide_x[:])
    # demand_y = np.array(demand_y)
    # y_train = demand_y[:, :, :].reshape(int((len(demand_data)-8)/4), 8, 8, 1)
    # # y_train = demand_y[0:9374]
    # x_valid.append(demand_x[:])
    # x_valid.append(supply_x[:])
    # x_valid.append(guide_x[:])
    # # y_valid = demand_y[9374:]
    # y_valid = demand_y[:, :, :].reshape(int((len(demand_data)-8)/4), 8, 8, 1)
    # print('========================')
    # print(len(y_valid))
    # print(y_valid.shape)
    # print('========================')
    demand_data = pd.read_pickle("./preprocessing_v2/total/"+region+"_DemandData_1step.pkl")
    # demand_data = pd.read_pickle("data/total/DemandData_normal.pkl") ## last used
    # demand_data = pd.read_pickle("data/total/DemandData_1step.pkl")
    demand_data = demand_data.T
    print('type of data',type(demand_data))
    print('length of data :', len(demand_data))

    mask = cv2.imread('./region/'+region+'filter.jpg',cv2.IMREAD_GRAYSCALE)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] > 127:
                mask[i][j] = 1
            else:
                mask[i][j] = 0
    # print(mask)
    for idx,eliment in enumerate(demand_data):
        demand_data[idx] = np.multiply(eliment,mask)
    # print(demand_data[0])
    demand_x = []
    demand_y = []
    tmp = []
    for idx,data in enumerate(demand_data):
        tmp.append(data)
        if len(tmp) == 4:
            demand_x.append(tmp)
            if idx != len(demand_data) - 1:
                # print(idx)
                demand_y.append(demand_data[idx + 1])
            else:
                demand_y.append(demand_data[idx])
            tmp = []

    # print(‘dmand_x_transpose’,np.array(demand_x))
    # print('demand_x:')
    demand_x = np.array(demand_x).swapaxes(1, 3)
    # print(demand_x.shape)
    # print('demand_y:')
    # print(np.array(demand_y).shape)
    supply_x = []
    # print('demand_x:',len(demand_x))
    # print('demand_data shape : ',demand_data.shape)
    for i in range(len(demand_x)):
        tmp = []
        # print('start at %s' %i)
        # if 8 + (i * 4) +8 > len(demand_x)-1:
        #     break
        for j in range(8):

            # print(8+(i*4)+j)
            if 8+(i*4)+j == len(demand_x):
                tmp.append(demand_data[8 + (i * 4) + j-1])
            else:
                tmp.append(demand_data[-1])
        # print('end')
        supply_x.append(tmp)
        tmp = []
        # supply_x.append(demand_data[8+(i * 4):(i * 4)])
        # print(‘appended : ’,demand_data[i*8:(i*8)+8])
        # print(‘appended : ’, demand_data[i * 8:(i * 8) + 8].shape)
    # print('supply_x:')
    supply_x = np.array(supply_x)
    # print(supply_x.shape)
    # supply_x = demand_x
    guide_x = []
    guide_data = pd.read_pickle('./preprocessing_v2/total/'+region+'_GuideData_1step.pkl')
    # guide_data = pd.read_pickle('data/total/GuideData_normal.pkl')
    guide_data = guide_data.T
    # print(guide_data)
    # print(guide_data[0])
    guide_x = guide_data[::4]
    guide_x = guide_x.reshape(len(guide_x),57)

    ## roi data ##
    roi_data = pd.read_pickle('./preprocessing_v2/total/'+region+'_ROIData_1step.pkl')
    # guide_data = pd.read_pickle('data/total/GuideData_normal.pkl')
    roi_input = copy.deepcopy(roi_data)
    roi_input = roi_input.T
    roi_input.reshape(len(roi_input), 57)
    roi_input1 = roi_input[0::4]
    roi_input2 = roi_input[1::4]
    roi_input3 = roi_input[2::4]
    roi_input4 = roi_input[3::4]

    roi_x = [x + y + z + w for x, y, z, w in zip(roi_input1, roi_input2, roi_input3, roi_input4)]
    roi_x.reshape(len(roi_x), 57)
    roi_x = np.array(roi_x)
    roi_x = roi_x.reshape(len(roi_x), 57)
    ## roi data ##

    # print('guide_x:')
    # print(np.array(guide_x).shape)
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    # x_train.append(np.concatenate((demand_x[:int(len(demand_x)*0.5)-1],demand_x[int(len(demand_x)*0.75):]),axis=0))
    # x_train.append(np.concatenate((supply_x[:int(len(supply_x)*0.5)-1],supply_x[int(len(supply_x)*0.75):]),axis=0))
    # x_train.append(np.concatenate((guide_x[:int(len(guide_x)*0.5)-1],guide_x[int(len(guide_x)*0.75):]),axis=0))
    x_train.append(demand_x[:int(len(demand_x)*0.8)])
    x_train.append(supply_x[:int(len(supply_x)*0.8)])
    x_train.append(guide_x[:int(len(guide_x)*0.8)])
    x_train.append(roi_x[:int(len(roi_x) * 0.8)])
    demand_y = np.array(demand_y)
    y_train = (demand_y[:int(len(demand_y)*0.8), :, :]).reshape(int(len(demand_y[:int(len(demand_y)*0.8), :, :])), 8, 8, 1)
    # print(int(len(demand_y)*0.75))
    # print('int(len(demand_y)*0.5)',int(len(demand_y)*0.5))
    # print('demand_y[:int(len(demand_y)*0.5), :, :]',len(demand_y[:int(len(demand_y)*0.5), :, :]))
    # print('demand_y[int(len(demand_y)*0.75):, :, :]',len(demand_y[int(len(demand_y)*0.75):, :, :]))
    # print('int(len(demand_y)*0.75)',int(len(demand_y)*0.75))
    # y_train = np.concatenate((demand_y[:int(len(demand_y)*0.5)-1, :, :],demand_y[int(len(demand_y)*0.75):, :, :]),axis=0).reshape(int(len(demand_y)*0.75), 8, 8, 1)
    # y_train = demand_y[0:9374]
    x_valid.append(demand_x[int(len(demand_x)*0.8):int(len(demand_x)*0.9)])
    x_valid.append(supply_x[int(len(supply_x)*0.8):int(len(supply_x)*0.9)])
    x_valid.append(guide_x[int(len(guide_x)*0.8):int(len(guide_x)*0.9)])
    x_valid.append(roi_x[int(len(roi_x) * 0.8):int(len(roi_x) * 0.9)])
    # y_valid = demand_y[9374:]
    # print('demand_y[int(len(demand_y)*0.5):int(len(demand_y)*0.75), :, :]',len(demand_y[int(len(demand_y)*0.5)-1:int(len(demand_y)*0.75), :, :]))
    y_valid = demand_y[int(len(demand_y)*0.8):int(len(demand_y) * 0.9), :, :].reshape(int(len(demand_y[int(len(demand_y)*0.8):int(len(demand_y) * 0.9), :, :])), 8, 8, 1)
    print('#### Data Loading Process End ####')
    print('Time is : ', datetime.datetime.now())
    return x_train, y_train, x_valid, y_valid

def min_max_calc(region):
    demand_data = pd.read_pickle("./preprocessing_v2/total/" + region + "_DemandData_1step.pkl")
    max = np.max(demand_data)
    min = np.min(demand_data)
    return max, min

# load_data('seoul')
# load_prediction_data('seoul')
load_test_data('seoul')