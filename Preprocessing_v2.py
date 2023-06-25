#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:38:19 2023

@author: kdml
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:08:25 2022
​
@author: kdml
"""
import json, requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import xml.etree.ElementTree as ET
import os
import glob, pickle
from datetime import datetime, date
import matplotlib.cm as cm
from tqdm import tqdm

# Json_Filelist = os.listdir('C:/Users/KUSRC/Desktop/Customized_TGNet/Customized_TGNet/raw_data/NIA_08_11_Total/')
def RegionCorditable(Cordtable, region):
    RegtionCordinatemap = {
        '서울특별시': {'Latitude_N': 37.715133, 'Latitude_S': 37.413294, 'Longitude_E': 127.269311,
                  'Longitude_W': 126.734086},
        '대전광역시': {'Latitude_N': 36.499767, 'Latitude_S': 36.182553, 'Longitude_E': 127.558644,
                  'Longitude_W': 127.247252},
        '경기도': {'Latitude_N': 38.300603, 'Latitude_S': 36.872260, 'Longitude_E': 127.830532,
                'Longitude_W': 126.262021},
        '대구광역시': {'Latitude_N': 36.016252, 'Latitude_S': 35.606077, 'Longitude_E': 128.762956,
                  'Longitude_W': 128.351221},
        '부산광역시': {'Latitude_N': 35.395936, 'Latitude_S': 34.879908, 'Longitude_E': 129.372819,
                  'Longitude_W': 128.738436},
        '울산광역시': {'Latitude_N': 35.737658, 'Latitude_S': 35.313041, 'Longitude_E': 129.484097,
                  'Longitude_W': 128.960786},
        '광주광역시': {'Latitude_N': 35.258435, 'Latitude_S': 35.050704, 'Longitude_E': 127.021811,
                  'Longitude_W': 126.645006},
        '인천광역시': {'Latitude_N': 37.831527, 'Latitude_S': 37.223039, 'Longitude_E': 126.792490,
                  'Longitude_W': 126.155716},
    }
    currentRegion = RegtionCordinatemap[region]
    X1 = (np.linspace(currentRegion['Longitude_W'], currentRegion['Longitude_E'], 9))
    Y1 = (np.linspace(currentRegion['Latitude_N'], currentRegion['Latitude_S'], 9))

    cordx = []
    cordy = []
    l = []
    c = []

    Xd = abs(X1[1] - X1[0])
    Yd = abs(Y1[1] - Y1[0])

    for i in range(0, 8):
        for j in range(0, 8):
            cordy.append((currentRegion['Latitude_S'] + Yd * (i + 0.5)))
            cordx.append((currentRegion['Longitude_W'] + Xd * (j + 0.5)))
            l.append(i)
            c.append(j)
    # plt.scatter(cordx, cordy)
    # plt.show()
    Cordtable['latitude'] = cordy
    Cordtable['longitude'] = cordx
    Cordtable['l'] = l
    Cordtable['c'] = c

    return Cordtable
def ReadJson(Json_Filelist, Regionlist, Monthpath, RoIdict):
    df_cols = ["name", "time", "latitude", "longitude"]
    Temp_MetaList = {i: np.zeros((57, 1, 24)) for i in Regionlist}
    Temp_ROIList = {i: np.zeros((57, 1, 24)) for i in Regionlist}

    DfList = {i: pd.DataFrame(columns=df_cols) for i in Regionlist}
    for jsonfile in Json_Filelist:
        with open(Monthpath + '/' + jsonfile, 'rt', encoding='UTF8') as st_json:
            data = json.load(st_json)

        if not (data['annotation'][0].get("latitude") == None and data['annotation'][0].get("longitude") == None):
            image_lat = round(float(data['annotation'][0]['latitude']), 5)
            image_lng = round(float(data['annotation'][0]['longitude']), 5)
            image_date = data['traffic_data']['createdDate']

            image_day = date(int(image_date[0:4]), int(image_date[4:6]), int(image_date[6:8])).weekday()
            if image_day >= 5:
                holiday = 1
            else:
                holiday = 0
            image_time = int(image_date[8:10])
            image_weather = int(data['app_weather']['app_weather_status'])
            if image_weather < 0:
                image_weather = 0
            if image_weather > 4:
                image_weather = 0
            result = data['user_info']['user_location']
            for reg in Regionlist:
                if result in reg:
                    result = reg

            if type(DfList.get(result)) != type(None):
                # if ((Latitude_S <= image_lat) and (image_lat <=Latitude_N)) and ((Longitude_W <= image_lng) and (image_lng <=Longitude_E)) :
                row = {
                    "name": image_date,
                    "time": image_time,
                    "latitude": image_lat,
                    "longitude": image_lng}

                DfList[result] = DfList[result].append(row, ignore_index=True)
                Temp_MetaList[result][image_time, 0, image_time] = 1
                Temp_MetaList[result][image_day + 24, 0, image_time] = 1
                Temp_MetaList[result][31, 0, image_time] = holiday
                Temp_MetaList[result][32 + image_weather, 0, image_time] = 1
                Temp_ROIList[result][RoIdict[data['annotation'][0]['label']], 0, image_time] += 1

    return DfList, Temp_MetaList, Temp_ROIList

def Calculate_distance(Cordtable, latitude, longitude):
    for temp in Cordtable.iterrows():
        co_latitude = temp[1]['latitude']
        co_longitude = temp[1]['longitude']
        Cordtable.loc[temp[0], 'distance'] = math.sqrt(
            (co_latitude - latitude) ** 2 + (co_longitude - longitude) ** 2)
    low = Cordtable.loc[Cordtable['distance'].argmin()]['l']
    column = Cordtable.loc[Cordtable['distance'].argmin()]['c']

    return Cordtable, low, column

def Preprocessiong_v2(path,test=False):
    ROIList = ['근린생활시설', '골프연습장', '판매시설', '대형판매시설', '위락시설',
               '관람집회시설', '의료시설', '교육연구시설', '운동시설', '숙박시설', '문화관람시설', '생태관람시설', '공장시설', '자동차관련시설', '방송통신시설', '관광휴게시설',
               '운수시설', '공항, 항만시설', '기타']
    RoIdict = {i: index for index, i in enumerate(ROIList)}

    # ResoucePath = 'dataset/'
    if test:
        OriginPath = path + 'test/origin/'
        ResoucePath = path + 'test/label/'
    else:
        OriginPath = path + 'total/origin/'
        ResoucePath = path + 'total/label/'

    Monthlist = os.listdir(ResoucePath)
    if '.DS_Store' in Monthlist:
        Monthlist.remove('.DS_Store')

    Monthlist.sort()
    Origin_cnt = 0
    Label_cnt = 0
    Matching_cnt = 0
    for Month in Monthlist:
        # print(Month,' 월 데이터 검증 시작')
        OriginFile = os.listdir(OriginPath+Month+'/')
        Origin_cnt += len(OriginFile)
        ResourceFile = os.listdir(ResoucePath +Month+'/')
        Label_cnt +=len(ResourceFile)
        for origin in OriginFile:
            for resource in ResourceFile:
                if origin[6:-5] in resource:
                    Matching_cnt +=1
        # print(Month, ' 월 데이터 검증 종료')
    # print('원천데이터 Json 파일 수 : ',Origin_cnt)
    # print('라벨링데이터 Json 파일 수 : ', Label_cnt)
    # print('원천 - 라벨링 매칭 데이터 수 : ', Matching_cnt)
    DementData = np.zeros(shape=(8, 8, 24))
    Regionlist = ['서울특별시', '대전광역시', '경기도', '대구광역시', '부산광역시', '울산광역시', '광주광역시', '인천광역시']

    RealDementData = {i: DementData for i in Regionlist}
    RealGuideData = {i: np.zeros((57, 1, 24)) for i in Regionlist}
    RealROIData = {i: np.zeros((57, 1, 24)) for i in Regionlist}

    colors = cm.rainbow(np.linspace(0, 1, 24))
    count = 0
    index = 0

    for Month in Monthlist:
        if test:
            Monthpath = ResoucePath + Month + '/'
        else:
            Monthpath = ResoucePath + Month + '/'
        JsonDateSplitList = {}

        JsonDataList = os.listdir(Monthpath)
        if '.DS_Store' in JsonDataList:
            JsonDataList.remove('.DS_Store')
        for i in JsonDataList:
            if i.split('_')[1].split('_')[0] in JsonDateSplitList.keys():
                JsonDateSplitList[i.split('_')[1].split('_')[0]].append(i)
            else:
                JsonDateSplitList[i.split('_')[1].split('_')[0]] = [i]
        DateKeylist = list(JsonDateSplitList.keys())
        DateKeylist.sort()
        for JsonDataList in DateKeylist:

            print('%s 일 json 파일 시계열 데이터 변환 작업 시작' % JsonDataList)

            TrafficData, metaData, ROIData = ReadJson(JsonDateSplitList[JsonDataList], Regionlist, Monthpath, RoIdict)

            for region in Regionlist:
                DementData = np.zeros(shape=(8, 8, 24))

                Cordtable = RegionCorditable(pd.DataFrame(columns=['latitude', 'longitude', 'l', 'c', ]), region)
                count += TrafficData[region].shape[0]
                y = Cordtable['latitude'].tolist()
                y1 = TrafficData[region]['latitude'].tolist()
                time = TrafficData[region]['time'].tolist()
                x = Cordtable['longitude'].tolist()
                x1 = TrafficData[region]['longitude'].tolist()

                for temp in TrafficData[region].iterrows():
                    latitude = temp[1]['latitude']
                    longitude = temp[1]['longitude']
                    time = temp[1]['time']

                    Cordtable, y, x = Calculate_distance(Cordtable, latitude, longitude)

                    DementData[int(y), int(x), time] += 1
                if index == 0:
                    RealDementData[region] = DementData
                    RealGuideData[region] = metaData[region]
                    RealROIData[region] = ROIData[region]

                else:
                    RealDementData[region] = np.concatenate((RealDementData[region], DementData), axis=2)
                    RealGuideData[region] = np.concatenate((RealGuideData[region], metaData[region]), axis=2)
                    RealROIData[region] = np.concatenate((RealROIData[region], ROIData[region]), axis=2)

            index += 1
            print('%d 번째 json 파일 시계열 데이터 변환 작업 끝' % index)

    # print("총 Json 파일 수 : ",len(Json_Filelist))
    # for index, json_f in enumerate(Json_Filelist):
    #     print('%d 번째 json 파일 시계열 데이터 변환 작업 시작' % index)
    #     print('파일명 : ',json_f)
    #     # TrafficData, metaData = ReadJson('C:/Users/KUSRC/Desktop/Customized_TGNet/Customized_TGNet/raw_data/NIA_08_11_Total/' + json_f, Regionlist)
    #
    #     TrafficData, metaData = ReadJson(Json_Filelist, Regionlist)
    #
    #     for region in Regionlist:
    #         Cordtable = RegionCorditable(pd.DataFrame(columns=['latitude', 'longitude', 'l', 'c', ]), region)
    #
    #         count += TrafficData[region].shape[0]
    #         y = Cordtable['latitude'].tolist()
    #         y1 = TrafficData[region]['latitude'].tolist()
    #         time = TrafficData[region]['time'].tolist()
    #         x = Cordtable['longitude'].tolist()
    #         x1 = TrafficData[region]['longitude'].tolist()
    #
    #         for temp in TrafficData[region].iterrows():
    #             latitude = temp[1]['latitude']
    #             longitude = temp[1]['longitude']
    #             time = temp[1]['time']
    #
    #             Cordtable, y, x = Calculate_distance(Cordtable, latitude, longitude)
    #
    #             DementData[int(y), int(x), time] += 1
    #         if index == 0:
    #             RealDementData[region] = DementData
    #             RealGuideData[region] = metaData[region]
    #         else:
    #             RealDementData[region] = np.concatenate((RealDementData[region], DementData), axis=2)
    #             RealGuideData[region] = np.concatenate((RealGuideData[region], metaData[region]), axis=2)
    #     print('%d 번째 json 파일 시계열 데이터 변환 작업 끝' % index)

    # with open(os.getcwd() + '/preprocessing/DemandData_1step.pkl', 'rb') as f:
    #     RealDementData = pickle.load(f)
    #
    # with open(os.getcwd() + '/preprocessing/GuideData_1step.pkl', 'rb') as f:
    #     RealGuideData = pickle.load(f)
    #
    #     RealDementData = np.concatenate((RealDementData, RealDementData), axis=2)
    #     RealGuideData = np.concatenate((RealGuideData, RealGuideData), axis=2)


    if test:
        print('테스트 시계열 데이터 Pickle 형식 파일 저장 작업 시작')
        with open('./preprocessing_v2/test/_DemandData_1step.pkl', 'wb') as f:
            pickle.dump(RealDementData, f)

        with open('./preprocessing_v2/test/_GuideData_1step.pkl', 'wb') as f:
            pickle.dump(RealGuideData, f)

        with open('./preprocessing_v2/test/_ROIData_1step.pkl', 'wb') as f:
            pickle.dump(RealROIData, f)
        print('전체 시계열 데이터 Pickle 형식 파일 저장 작업 끝')

        for reg in Regionlist:
            print(reg + '테스트 시계열 데이터 Pickle 형식 파일 저장 작업 시작')
            # print(RealDementData[reg])
            # region_maximum_value = np.max(RealDementData[reg][:, :, :])
            # for i in range(0, RealDementData[reg].shape[2]):
            #     RealDementData[reg][:, :, i] = RealDementData[reg]
            if reg == "서울특별시":
                trans_reg = 'seoul'
            elif reg == "대전광역시":
                trans_reg = 'daejeon'
            elif reg == "경기도":
                trans_reg = 'gyeonggi'
            elif reg == '대구광역시':
                trans_reg = 'daegu'
            elif reg == '부산광역시':
                trans_reg = 'busan'
            elif reg == '울산광역시':
                trans_reg = 'ulsan'
            elif reg == '광주광역시':
                trans_reg = 'gwangju'
            elif reg == '인천광역시':
                trans_reg = 'incheon'
            else:
                trans_reg = reg

            with open('./preprocessing_v2/test/' + trans_reg + '_DemandData_1step.pkl','wb') as f:
                pickle.dump(RealDementData[reg], f)
            with open('./preprocessing_v2/test/' + trans_reg + '_GuideData_1step.pkl','wb') as f:
                pickle.dump(RealGuideData[reg], f)
            with open('./preprocessing_v2/test/' + trans_reg + '_ROIData_1step.pkl','wb') as f:
                pickle.dump(RealROIData[reg], f)
            print(reg + '지역 시계열 데이터 Pickle 형식 파일 저장 작업 끝')
    else:
        print('전체 시계열 데이터 Pickle 형식 파일 저장 작업 시작')
        with open('./preprocessing_v2/total/_DemandData_1step.pkl', 'wb') as f:
            pickle.dump(RealDementData, f)

        with open('./preprocessing_v2/total/_GuideData_1step.pkl', 'wb') as f:
            pickle.dump(RealGuideData, f)

        with open('./preprocessing_v2/total/_ROIData_1step.pkl', 'wb') as f:
            pickle.dump(RealROIData, f)
        print('전체 시계열 데이터 Pickle 형식 파일 저장 작업 끝')

        for reg in Regionlist:
            print(reg + '지역 시계열 데이터 Pickle 형식 파일 저장 작업 시작')
            # print(RealDementData[reg])
            # region_maximum_value = np.max(RealDementData[reg][:, :, :])
            # for i in range(0, RealDementData[reg].shape[2]):
            #     RealDementData[reg][:, :, i] = RealDementData[reg]
            if reg == "서울특별시":
                trans_reg = 'seoul'
            elif reg == "대전광역시":
                trans_reg = 'daejeon'
            elif reg == "경기도":
                trans_reg = 'gyeonggi'
            elif reg == '대구광역시':
                trans_reg = 'daegu'
            elif reg == '부산광역시':
                trans_reg = 'busan'
            elif reg == '울산광역시':
                trans_reg = 'ulsan'
            elif reg == '광주광역시':
                trans_reg = 'gwangju'
            elif reg == '인천광역시':
                trans_reg = 'incheon'
            else:
                trans_reg = reg

            with open('./preprocessing_v2/total/' + trans_reg + '_DemandData_1step.pkl','wb') as f:
                pickle.dump(RealDementData[reg], f)
            with open('./preprocessing_v2/total/' + trans_reg + '_GuideData_1step.pkl','wb') as f:
                pickle.dump(RealGuideData[reg], f)
            with open('./preprocessing_v2/total/' + trans_reg + '_ROIData_1step.pkl','wb') as f:
                pickle.dump(RealROIData[reg], f)
            print(reg + '지역 시계열 데이터 Pickle 형식 파일 저장 작업 끝')

        # with open(os.getcwd() + '/preprocessing/'+reg+'_DemandData_1step.pkl', 'rb') as f:
        #    RealDementData = pickle.load(f)

        # with open(os.getcwd() + '/preprocessing/'+reg+'_GuideData_1step.pkl', 'rb') as f:
        #    RealGuideData = pickle.load(f)

        # with open(os.getcwd() + '/preprocessing/'+reg+'_DemandData_1step.pkl', 'rb') as f:
        #    DementData = pickle.load(f)

        # with open(os.getcwd() + '/preprocessing/'+reg+'_GuideData_1step.pkl', 'rb') as f:
        #    GuideData = pickle.load(f)

    print('data save task end')
    # maximum_value = np.max(RealDementData[:, :, :])
    # for i in range(0, RealDementData.shape[2]):
    #     RealDementData[:, :, i] = RealDementData[:, :, i] / maximum_value
    #
    # with open(os.getcwd() + '/preprocessing/DemandData_1step.pkl', 'wb') as f:
    #     pickle.dump(RealDementData, f)
    #
    # with open(os.getcwd() + '/preprocessing/GuideData_1step.pkl', 'wb') as f:
    #     pickle.dump(RealGuideData, f)
    #
    # with open(os.getcwd() + '/preprocessing/DemandData_1step.pkl', 'rb') as f:
    #     RealDementData = pickle.load(f)
    #
    # with open(os.getcwd() + '/preprocessing/GuideData_1step.pkl', 'rb') as f:
    #     RealGuideData = pickle.load(f)
    #
    # with open(os.getcwd() + '/preprocessing/DemandData_1step.pkl', 'rb') as f:
    #     DementData = pickle.load(f)
    #
    # with open(os.getcwd() + '/preprocessing/GuideData_1step.pkl', 'rb') as f:
    #     GuideData = pickle.load(f)
    #
    # print('data save task end')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--data_path',type=str,default='./dataset/')
    args = parser.parse_args()
    Preprocessiong_v2(args.data_path, args.test)
