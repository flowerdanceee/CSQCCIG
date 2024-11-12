from pytorch_fid import fid_score
import torch
from torchvision import transforms

transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
# real
real_path0 = 'real-mnist/0'
real_path1 = 'real-mnist/1'

# quantum(P)
P1L36_path0 = 'quantum_fid/P=1,L=36/0'
P1L36_path1 = 'quantum_fid/P=1,L=36/1'
P2L19_path0 = 'quantum_fid/P=2,L=19/0'
P2L19_path1 = 'quantum_fid/P=2,L=19/1'
P4L11_path0 = 'quantum_fid/P=4,L=11/0'
P4L11_path1 = 'quantum_fid/P=4,L=11/1'
P7L7_path0 = 'quantum_fid/P=7,L=7/0'
P7L7_path1 = 'quantum_fid/P=7,L=7/1'
P8L6_path0 = 'quantum_fid/P=8,L=6/0'
P8L6_path1 = 'quantum_fid/P=8,L=6/1'
P14L4_path0 = 'quantum_fid/P=14,L=4/0'
P14L4_path1 = 'quantum_fid/P=14,L=4/1'
P98L1_path0 = 'quantum_fid/P=98,L=1/0'
P98L1_path1 = 'quantum_fid/P=98,L=1/1'

# quantum(L)
P8L5_path0 = 'quantum_fid/P=8,L=5/0'
P8L5_path1 = 'quantum_fid/P=8,L=5/1'
P8L4_path0 = 'quantum_fid/P=8,L=4/0'
P8L4_path1 = 'quantum_fid/P=8,L=4/1'

# quantum(P)
fid_value0 = fid_score.calculate_fid_given_paths([real_path0, P1L36_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, P1L36_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('P1L36:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, P2L19_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, P2L19_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('P2L19:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, P4L11_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, P4L11_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('P4L11:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, P7L7_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, P7L7_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('P7L7:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, P8L6_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, P8L6_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('P8L6:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, P14L4_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, P14L4_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('P14L4:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, P98L1_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, P98L1_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('P98L1:', fid_value)

# quantum(L)
fid_value0 = fid_score.calculate_fid_given_paths([real_path0, P8L5_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, P8L5_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('P8L5:', fid_value)

fid_value0 = fid_score.calculate_fid_given_paths([real_path0, P8L4_path0], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value1 = fid_score.calculate_fid_given_paths([real_path1, P8L4_path1], batch_size=64,
                                                 device=torch.device('cuda'), dims=2048, num_workers=1)
fid_value = (fid_value1 + fid_value0) / 2
print('P8L4:', fid_value)
