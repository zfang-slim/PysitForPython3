Dir = '/scratch/Zhilong/ExxonProject/ConstantModel/ExxonMeeting20190314/OneEvent_SF_PGD_Ormsby/Exp7_5/EIP1/xm';
file = [Dir,'/x_99.mat'];
x = load(file);
x = x.data;
x = reshape(x,201,101);
imagesc(x');colormap(jet);caxis([2,3])

file = [Dir,'/x_0.mat'];
x0 = load(file);
x0 = x0.data;
x0 = reshape(x0,201,101);
figure;imagesc(x'-x0');caxis([-.1,.1]);colormap(redblue)
figure;imagesc(x0');caxis([2,3]);colormap(jet)