python Preprocessing_v2.py --test True
python main.py --num_gpu 0 --model_name busan --region busan --scale min_max --test | tee ./log/busan모델_테스트_터미널_로그.txt
python main.py --num_gpu 0 --model_name daegu --region daegu --scale min_max --test | tee ./log/daegu모델_테스트_터미널_로그.txt
python main.py --num_gpu 0 --model_name daejeon --region daejeon --scale min_max --test | tee ./log/daejeon모델_테스트_터미널_로그.txt
python main.py --num_gpu 0 --model_name gwangju --region gwangju --scale min_max --test | tee ./log/gwangju모델_테스트_터미널_로그.txt
python main.py --num_gpu 0 --model_name gyeonggi --region gyeonggi --scale min_max --test | tee ./log/gyeonggi모델_테스트_터미널_로그.txt
python main.py --num_gpu 0 --model_name incheon --region incheon --scale min_max --test | tee ./log/incheon모델_테스트_터미널_로그.txt
python main.py --num_gpu 0 --model_name seoul --region seoul --scale min_max --test | tee ./log/seoul모델_테스트_터미널_로그.txt
python main.py --num_gpu 0 --model_name ulsan --region ulsan --scale min_max --test | tee ./log/ulsan모델_테스트_터미널_로그.txt
python generate_log.py --region busan
python generate_log.py --region daegu
python generate_log.py --region daejeon
python generate_log.py --region gwangju
python generate_log.py --region gyeonggi
python generate_log.py --region incheon
python generate_log.py --region seoul
python generate_log.py --region ulsan