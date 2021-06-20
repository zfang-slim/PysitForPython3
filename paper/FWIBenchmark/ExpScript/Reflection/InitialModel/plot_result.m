Dir = '/scratch/Zhilong/ExxonProject/ReflectModel/ExxonMeeting201909/Reflector_PQN2/Exp2_5/EIP1/xm';
file = [Dir,'/x_40.mat'];
x = load(file);
x = x.data;
x = reshape(x,101,301);
figure;imagesc(x);colormap(jet);caxis([2,2.6])

file = [Dir,'/x_0.mat'];
x0 = load(file);
x0 = x0.data;
x0 = reshape(x0,101,301);
figure;imagesc(x-x0);colormap(jet);caxis([-.5,.5]);colormap(redblue)
figure;imagesc(x0);colormap(jet);caxis([2,2.6])