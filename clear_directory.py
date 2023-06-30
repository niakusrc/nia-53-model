import os
import datetime
import shutil
def clear_directory(dir_name):
    dir_path = os.getcwd()
    # print(dir_path)
    if os.path.isdir(dir_path + '/' + dir_name):
        print('프로젝트 경로 ' + dir_path + '/' + '에 존재하는 폴더"' + dir_name + '"를 삭제합니다.')
        shutil.rmtree(dir_path + '/' + dir_name)
        print('프로젝트 경로 ' + dir_path + '/' + '에 존재하는 폴더"' + dir_name + '"를 삭제하였습니다.')
        os.makedirs(dir_path + '/' + dir_name)
        with open(dir_path + '/' + dir_name + '/' +dir_name, 'w') as f:
            f.write(dir_name)



if __name__ == "__main__":
    import argparse
    print('[*] Program Starts')
    print('Time is : ', datetime.datetime.now())
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dir_name',type=str)
    args = parser.parse_args()
    print('[*] Command : python clear_directory.py --dir_name ' + str(args.dir_name))
    print('[!] Clear Directory Start')
    print('Time is : ', datetime.datetime.now())
    clear_directory(args.dir_name)
    print('[*] Clear Directory End')
    print('Time is : ', datetime.datetime.now())
