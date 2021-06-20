Dir = '/scratch/Zhilong/ExxonProject/ConstantModel/ExxonMeeting2020/OneEvent_PGD/Exp2_67/EIP1';
file = [Dir,'/x_100.mat'];
x = load(file);
x = x.data;
x = reshape(x,201,101);
imagesc(x');colormap(jet);caxis([2,3])

file = [Dir,'/x_0.mat'];
x0 = load(file);
x0 = x0.data;
x0 = reshape(x0,201,101);
figure;imagesc(x'-x0');colormap(jet);caxis([-.5,.5]);colormap(redblue)