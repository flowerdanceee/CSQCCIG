from pytorch_fid import fid_score
import torch
from torchvision import transforms

transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
# real
real_path0 = 'real-mnist/0'
real_path1 = 'real-mnist/1'

# CNN
CNNc2_path0 = 'CNN_fid/c=2/0'
CNNc2_path1 = 'CNN_fid/c=2/1'
CNNc3_path0 = 'CNN_fid/c=3/0'
CNNc3_path1 = 'CNN_fid/c=3/1'
CNNc4_path0 = 'CNN_fid/c=4/0'
CNNc4_path1 = 'CNN_fid/c=4/1'
CNNc5_path0 = 'CNN_fid/c=5/0'
CNNc5_path1 = 'CNN_fid/c=5/1'
CNNc6_path0 = 'CNN_fid/c=6/0'
CNNc6_path1 = 'CNN_fid/c=6/1'
CNNc7_path0 = 'CNN_fid/c=7/0'
CNNc7_path1 = 'CNN_fid/c=7/1'
CNNc8_path0 = 'CNN_fid/c=8/0'
CNNc8_path1 = 'CNN_fid/c=8/1'

# MLP
MLPn4_path0 = 'MLP_fid/n=4/0'
MLPn4_path1 = 'MLP_fid/n=4/1'
MLPn5_path0 = 'MLP_fid/n=5/0'
MLPn5_path1 = 'MLP_fid/n=5/1'
MLPn6_path0 = 'MLP_fid/n=6/0'
MLPn6_path1 = 'MLP_fid/n=6/1'
MLPn7_path0 = 'MLP_fid/n=7/0'
MLPn7_path1 = 'MLP_fid/n=7/1'
MLPn8_path0 = 'MLP_fid/n=8/0'
MLPn8_path1 = 'MLP_fid/n=8/1'

# quantum
QP8L6_path0 = 'quantum_fid/P=8,L=6/0'
QP8L6_path1 = 'quantum_fid/P=8,L=6/1'

# CNN
fid_value0 = fid_score.calculate_fid_given_paths([real_path0, CNNc2_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, CNNc2_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('CNNc2:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, CNNc3_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, CNNc3_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('CNNc3:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, CNNc4_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, CNNc4_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('CNNc4:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, CNNc5_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, CNNc5_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('CNNc5:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, CNNc6_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, CNNc6_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('CNNc6:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, CNNc7_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, CNNc7_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('CNNc7:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, CNNc8_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, CNNc8_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('CNNc8:', fid_value)

# MLP
fid_value0 = fid_score.calculate_fid_given_paths([real_path0, MLPn4_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, MLPn4_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('MLPn4:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, MLPn5_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, MLPn5_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('MLPn5:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, MLPn6_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, MLPn6_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('MLPn6:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, MLPn7_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, MLPn7_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('MLPn7:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, MLPn8_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, MLPn8_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('MLPn8:', fid_value)

# quantum
fid_value0 = fid_score.calculate_fid_given_paths([real_path0, QP8L6_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, QP8L6_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('QP8L6:', fid_value)
