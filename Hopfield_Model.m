clc
clear all
close all
load lab2_2_data.mat
%% Storage Phase (Learning)
N= length(p0);
w0=p0*p0';
w1=p1*p1';
w2=p2*p2';
W= 1/N *(w0+w1+w2);
Weight_mat= W- diag(diag(W));
%% Distorting images with different levels of 0.05, 0.1, and 0.25
d_0_1=distort_image(p0,0.05);
d_1_1=distort_image(p1,0.05);
d_2_1=distort_image(p2,0.05);

d_0_2=distort_image(p0,0.1);
d_1_2=distort_image(p1,0.1);
d_2_2=distort_image(p2,0.1);

d_0_3=distort_image(p0,0.25);
d_1_3=distort_image(p1,0.25);
d_2_3=distort_image(p2,0.25);
%% Initialization
epoch=10;
bias=0.5; %found with trial and error
%% d_0_1_recall phase
iter=1;
x_0_1 = d_0_1; %distorted version of 0
y_0_1=x_0_1;
while   (iter<=epoch)
    shuffled=randperm(N);
        for j=1:N
          rand_idx=shuffled(1,j);
          y_0_1(rand_idx,1)= sign((Weight_mat(rand_idx,:)*x_0_1)+bias);  %%update the state for this neuron
          aux=Weight_mat*y_0_1;
          energy_0(:,iter)=(-1/2*aux'*p0)-(bias*sum(y_0_1));
          energy_1(:,iter)=(-1/2 *aux'*p1)-(bias*sum(y_0_1));
          energy_2(:,iter)=(-1/2 *aux'*p2)-(bias*sum(y_0_1));

          m_0(:,iter)=1/N *(y_0_1'*p0);
          m_1(:,iter)=1/N *(y_0_1'*p1);
          m_2(:,iter)=1/N *(y_0_1'*p2);
        end
    if(isequal(y_0_1,x_0_1)) 
		break;
	end
        iter=iter+1;  
        x_0_1=y_0_1;
end

acc_0_1 = (1- 1/N*sum(y_0_1~=p0))*100;
figure;
subplot(2,1,1)
imagesc(reshape(d_0_1,32,32));
title('Distorted Image, d-0-1')
hold on
subplot(2,1,2)
imagesc(reshape(y_0_1,32,32))
title(sprintf('%s %.2f %s','reconstructed image,',acc_0_1,'%'))
% saveas(gcf,'distorted_0_1_reconstructed.png')

figure;
plot (1:iter, energy_0)
hold on
plot (1:iter, energy_1)
hold on
plot (1:iter, energy_2)
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Energy-0','Energy-1','Energy-2')
title('Energy Functions Related to d-0-1')
% saveas(gcf,'distorted_0_1_energy.png')

figure;
plot (1:iter, m_0)
hold on
plot (1:iter, m_1)
hold on
plot (1:iter, m_2)
set(gca,'YLim',[0 1])
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Overlap-0','Overlap-1','Overlap-2')
title('Overlap Functions Related to d-0-1')
% saveas(gcf,'distorted_0_1_overlap.png')


clear aux energy_0 energy_1 energy_2 m_0 m_1 m_2
%% d_1_1_recall phase
iter=1;
x_1_1 = d_1_1; %distorted version of 0
y_1_1=x_1_1;
while   (iter<=epoch)
    shuffled=randperm(N);
        for j=1:N
          rand_idx=shuffled(1,j);
          y_1_1(rand_idx,1)= sign((Weight_mat(rand_idx,:)*x_1_1)+bias);  %%update the state for this neuron
          aux=Weight_mat*y_1_1;
          energy_0(:,iter)=(-1/2*aux'*p0)-(bias*sum(y_1_1));
          energy_1(:,iter)=(-1/2 *aux'*p1)-(bias*sum(y_1_1));
          energy_2(:,iter)=(-1/2 *aux'*p2)-(bias*sum(y_1_1));

          m_0(:,iter)=1/N *(y_1_1'*p0);
          m_1(:,iter)=1/N *(y_1_1'*p1);
          m_2(:,iter)=1/N *(y_1_1'*p2);
        end
    if(isequal(y_1_1,x_1_1)) %dovrebbe arrestarsi in poche iterazioni
		break;
	end
        iter=iter+1;  
        x_1_1=y_1_1;
end

acc_1_1 = (1- 1/N*sum(y_1_1~=p1))*100;
figure;
subplot(2,1,1)
imagesc(reshape(d_1_1,32,32));
title('Distorted Image, d-1-1')
hold on
subplot(2,1,2)
imagesc(reshape(y_1_1,32,32))
title(sprintf('%s %.2f %s','reconstructed image,',acc_1_1,'%'))
% saveas(gcf,'distorted_1_1_reconstructed.png')

figure;
plot (1:iter, energy_0)
hold on
plot (1:iter, energy_1)
hold on
plot (1:iter, energy_2)
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Energy-0','Energy-1','Energy-2')
title('Energy Functions Related to d-0-1')
% saveas(gcf,'distorted_1_1_energy.png')

figure;
plot (1:iter, m_0)
hold on
plot (1:iter, m_1)
hold on
plot (1:iter, m_2)
set(gca,'YLim',[0 1])
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Overlap-0','Overlap-1','Overlap-2')
title('Overlap Functions Related to d-1-1')
% saveas(gcf,'distorted_1_1_overlap.png')


clear aux energy_0 energy_1 energy_2 m_0 m_1 m_2
%% d_2_1_recall phaseiter=1;
iter=1;
x_2_1 = d_2_1; %distorted version of 0
y_2_1=x_2_1;
while   (iter<=epoch)
    shuffled=randperm(N);
        for j=1:N
          rand_idx=shuffled(1,j);
          y_2_1(rand_idx,1)= sign((Weight_mat(rand_idx,:)*x_2_1)+bias);  %%update the state for this neuron
          aux=Weight_mat*y_2_1;
          energy_0(:,iter)=(-1/2*aux'*p0)-(bias*sum(y_2_1));
          energy_1(:,iter)=(-1/2 *aux'*p1)-(bias*sum(y_2_1));
          energy_2(:,iter)=(-1/2 *aux'*p2)-(bias*sum(y_2_1));

          m_0(:,iter)=1/N *(y_2_1'*p0);
          m_1(:,iter)=1/N *(y_2_1'*p1);
          m_2(:,iter)=1/N *(y_2_1'*p2);
        end
    if(isequal(y_2_1,x_2_1)) %dovrebbe arrestarsi in poche iterazioni
		break;
	end
        iter=iter+1;  
        x_2_1=y_2_1;
end

acc_2_1 = (1- 1/N*sum(y_2_1~=p2))*100;
figure;
subplot(2,1,1)
imagesc(reshape(d_2_1,32,32));
title('Distorted Image, d-2-1')
hold on
subplot(2,1,2)
imagesc(reshape(y_2_1,32,32))
title(sprintf('%s %.2f %s','reconstructed image,',acc_2_1,'%'))
% saveas(gcf,'distorted_2_1_reconstructed.png')

figure;
plot (1:iter, energy_0)
hold on
plot (1:iter, energy_1)
hold on
plot (1:iter, energy_2)
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Energy-0','Energy-1','Energy-2')
title('Energy Functions Related to d-2-1')
% saveas(gcf,'distorted_2_1_energy.png')

figure;
plot (1:iter, m_0)
hold on
plot (1:iter, m_1)
hold on
plot (1:iter, m_2)
set(gca,'YLim',[0 1])
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Overlap-0','Overlap-1','Overlap-2')
title('Overlap Functions Related to d-2-1')
% saveas(gcf,'distorted_2_1_overlap.png')


clear aux energy_0 energy_1 energy_2 m_0 m_1 m_2
%% d_0_2_recall phase
iter=1;
x_0_2 = d_0_2; %distorted version of 0
y_0_2=x_0_2;
while   (iter<=epoch) 
    shuffled=randperm(N);
        for j=1:N
          rand_idx=shuffled(1,j);
          y_0_2(rand_idx,1)= sign((Weight_mat(rand_idx,:)*x_0_2)+bias);  %%update the state for this neuron
          aux=Weight_mat*y_0_2;
          energy_0(:,iter)=(-1/2*aux'*p0)-(bias*sum(y_0_2));
          energy_1(:,iter)=(-1/2 *aux'*p1)-(bias*sum(y_0_2));
          energy_2(:,iter)=(-1/2 *aux'*p2)-(bias*sum(y_0_2));

          m_0(:,iter)=1/N *(y_0_2'*p0);
          m_1(:,iter)=1/N *(y_0_2'*p1);
          m_2(:,iter)=1/N *(y_0_2'*p2);
        end
    if(isequal(y_0_2,x_0_2)) %dovrebbe arrestarsi in poche iterazioni
		break;
	end
        iter=iter+1;  
        x_0_2=y_0_2;
end

acc_0_2 = (1- 1/N*sum(y_0_2~=p0))*100;
figure;
subplot(2,1,1)
imagesc(reshape(d_0_2,32,32));
title('Distorted Image, d-0-2')
hold on
subplot(2,1,2)
imagesc(reshape(y_0_2,32,32))
title(sprintf('%s %.2f %s','reconstructed image,',acc_0_2,'%'))
% saveas(gcf,'distorted_0_2_reconstructed.png')

figure;
plot (1:iter, energy_0)
hold on
plot (1:iter, energy_1)
hold on
plot (1:iter, energy_2)
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Energy-0','Energy-1','Energy-2')
title('Energy Functions Related to d-0-2')
% saveas(gcf,'distorted_0_2_energy.png')

figure;
plot (1:iter, m_0)
hold on
plot (1:iter, m_1)
hold on
plot (1:iter, m_2)
set(gca,'YLim',[0 1])
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Overlap-0','Overlap-1','Overlap-2')
title('Overlap Functions Related to d-0-2')
% saveas(gcf,'distorted_0_2_overlap.png')


clear aux energy_0 energy_1 energy_2 m_0 m_1 m_2
%% d_1_2_recall phase
iter=1;
x_1_2 = d_1_2; %distorted version of 0
y_1_2=x_1_2;
while   (iter<=epoch)
    shuffled=randperm(N);
        for j=1:N
          rand_idx=shuffled(1,j);
          y_1_2(rand_idx,1)= sign((Weight_mat(rand_idx,:)*x_1_2)+bias);  %%update the state for this neuron
          aux=Weight_mat*y_1_2;
          energy_0(:,iter)=(-1/2*aux'*p0)-(bias*sum(y_1_2));
          energy_1(:,iter)=(-1/2 *aux'*p1)-(bias*sum(y_1_2));
          energy_2(:,iter)=(-1/2 *aux'*p2)-(bias*sum(y_1_2));

          m_0(:,iter)=1/N *(y_1_2'*p0);
          m_1(:,iter)=1/N *(y_1_2'*p1);
          m_2(:,iter)=1/N *(y_1_2'*p2);
        end
    if(isequal(y_1_2,x_1_2)) %dovrebbe arrestarsi in poche iterazioni
		break;
	end
        iter=iter+1;  
        x_1_2=y_1_2;
end

acc_1_2 = (1- 1/N*sum(y_1_2~=p1))*100;
figure;
subplot(2,1,1)
imagesc(reshape(d_1_2,32,32));
title('Distorted Image, d-1-2')
hold on
subplot(2,1,2)
imagesc(reshape(y_1_2,32,32))
title(sprintf('%s %.2f %s','reconstructed image,',acc_1_2,'%'))
% saveas(gcf,'distorted_1_2_reconstructed.png')

figure;
plot (1:iter, energy_0)
hold on
plot (1:iter, energy_1)
hold on
plot (1:iter, energy_2)
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Energy-0','Energy-1','Energy-2')
title('Energy Functions Related to d-1-2')
% saveas(gcf,'distorted_1_2_energy.png')

figure;
plot (1:iter, m_0)
hold on
plot (1:iter, m_1)
hold on
plot (1:iter, m_2)
set(gca,'YLim',[0 1])
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Overlap-0','Overlap-1','Overlap-2')
title('Overlap Functions Related to d-1-2')
% saveas(gcf,'distorted_1_2_overlap.png')


clear aux energy_0 energy_1 energy_2 m_0 m_1 m_2
%% d_2_2_recall phase
iter=1;
x_2_2 = d_2_2; %distorted version of 0
y_2_2=x_2_2;
while   (iter<=epoch)
    shuffled=randperm(N);
        for j=1:N
          rand_idx=shuffled(1,j);
          y_2_2(rand_idx,1)= sign((Weight_mat(rand_idx,:)*x_2_2)+bias);  %%update the state for this neuron
          aux=Weight_mat*y_2_2;
          energy_0(:,iter)=(-1/2*aux'*p0)-(bias*sum(y_2_2));
          energy_1(:,iter)=(-1/2 *aux'*p1)-(bias*sum(y_2_2));
          energy_2(:,iter)=(-1/2 *aux'*p2)-(bias*sum(y_2_2));

          m_0(:,iter)=1/N *(y_2_2'*p0);
          m_1(:,iter)=1/N *(y_2_2'*p1);
          m_2(:,iter)=1/N *(y_2_2'*p2);
        end
    if(isequal(y_2_2,x_2_2)) %dovrebbe arrestarsi in poche iterazioni
		break;
	end
        iter=iter+1;  
        x_2_2=y_2_2;
end

acc_2_2 = (1- 1/N*sum(y_2_2~=p2))*100;
figure;
subplot(2,1,1)
imagesc(reshape(d_2_2,32,32));
title('Distorted Image, d-2-2')
hold on
subplot(2,1,2)
imagesc(reshape(y_2_2,32,32))
title(sprintf('%s %.2f %s','reconstructed image,',acc_2_2,'%'))
% saveas(gcf,'distorted_2_2_reconstructed.png')

figure;
plot (1:iter, energy_0)
hold on
plot (1:iter, energy_1)
hold on
plot (1:iter, energy_2)
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Energy-0','Energy-1','Energy-2')
title('Energy Functions Related to d-2-2')
% saveas(gcf,'distorted_2_2_energy.png')

figure;
plot (1:iter, m_0)
hold on
plot (1:iter, m_1)
hold on
plot (1:iter, m_2)
set(gca,'YLim',[0 1])
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Overlap-0','Overlap-1','Overlap-2')
title('Overlap Functions Related to d-2-2')
% saveas(gcf,'distorted_2_2_overlap.png')


clear aux energy_0 energy_1 energy_2 m_0 m_1 m_2
%% d_0_3_recall phase
iter=1;
x_0_3 = d_0_3; %distorted version of 0
y_0_3=x_0_3;
while   (iter<=epoch)
    shuffled=randperm(N);
        for j=1:N
          rand_idx=shuffled(1,j);
          y_0_3(rand_idx,1)= sign((Weight_mat(rand_idx,:)*x_0_3)+bias);  %%update the state for this neuron
          aux=Weight_mat*y_0_3;
          energy_0(:,iter)=(-1/2*aux'*p0)-(bias*sum(y_0_3));
          energy_1(:,iter)=(-1/2 *aux'*p1)-(bias*sum(y_0_3));
          energy_2(:,iter)=(-1/2 *aux'*p2)-(bias*sum(y_0_3));

          m_0(:,iter)=1/N *(y_0_3'*p0);
          m_1(:,iter)=1/N *(y_0_3'*p1);
          m_2(:,iter)=1/N *(y_0_3'*p2);
        end
    if(isequal(y_0_3,x_0_3)) %dovrebbe arrestarsi in poche iterazioni
		break;
	end
        iter=iter+1;  
        x_0_3=y_0_3;
end

acc_0_3 = (1- 1/N*sum(y_0_3~=p0))*100;
figure;
subplot(2,1,1)
imagesc(reshape(d_0_3,32,32));
title('Distorted Image, d-0-3')
hold on
subplot(2,1,2)
imagesc(reshape(y_0_3,32,32))
title(sprintf('%s %.2f %s','reconstructed image,',acc_0_3,'%'))
% saveas(gcf,'distorted_0_3_reconstructed.png')

figure;
plot (1:iter, energy_0)
hold on
plot (1:iter, energy_1)
hold on
plot (1:iter, energy_2)
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Energy-0','Energy-1','Energy-2')
title('Energy Functions Related to d-0-3')
% saveas(gcf,'distorted_0_3_energy.png')

figure;
plot (1:iter, m_0)
hold on
plot (1:iter, m_1)
hold on
plot (1:iter, m_2)
set(gca,'YLim',[0 1])
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Overlap-0','Overlap-1','Overlap-2')
title('Overlap Functions Related to d-0-3')
% saveas(gcf,'distorted_0_3_overlap.png')


clear aux energy_0 energy_1 energy_2 m_0 m_1 m_2
%% d_1_3_recall phase
iter=1;
x_1_3 = d_1_3; %distorted version of 1
y_1_3=x_1_3;
while   (iter<=epoch)
    shuffled=randperm(N);
        for j=1:N
          rand_idx=shuffled(1,j);
          y_1_3(rand_idx,1)= sign((Weight_mat(rand_idx,:)*x_1_3)+bias);  %%update the state for this neuron
          aux=Weight_mat*y_1_3;
          energy_0(:,iter)=(-1/2*aux'*p0)-(bias*sum(y_1_3));
          energy_1(:,iter)=(-1/2 *aux'*p1)-(bias*sum(y_1_3));
          energy_2(:,iter)=(-1/2 *aux'*p2)-(bias*sum(y_1_3));

          m_0(:,iter)=1/N *(y_1_3'*p0);
          m_1(:,iter)=1/N *(y_1_3'*p1);
          m_2(:,iter)=1/N *(y_1_3'*p2);
        end
    if(isequal(y_1_3,x_1_3)) %dovrebbe arrestarsi in poche iterazioni
		break;
	end
        iter=iter+1;  
        x_1_3=y_1_3;
end

acc_1_3 = (1- 1/N*sum(y_1_3~=p1))*100;
figure;
subplot(2,1,1)
imagesc(reshape(d_1_3,32,32));
title('Distorted Image, d-1-3')
hold on
subplot(2,1,2)
imagesc(reshape(y_1_3,32,32))
title(sprintf('%s %.2f %s','reconstructed image,',acc_1_3,'%'))
% saveas(gcf,'distorted_1_3_reconstructed.png')

figure;
plot (1:iter, energy_0)
hold on
plot (1:iter, energy_1)
hold on
plot (1:iter, energy_2)
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Energy-0','Energy-1','Energy-2')
title('Energy Functions Related to d-1-3')
% saveas(gcf,'distorted_1_3_energy.png')

figure;
plot (1:iter, m_0)
hold on
plot (1:iter, m_1)
hold on
plot (1:iter, m_2)
set(gca,'YLim',[0 1])
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Overlap-0','Overlap-1','Overlap-2')
title('Overlap Functions Related to d-1-3')
% saveas(gcf,'distorted_1_3_overlap.png')


clear aux energy_0 energy_1 energy_2 m_0 m_1 m_2
%% d_2_3_recall phase
iter=1;
x_2_3 = d_2_3; %distorted version of 0
y_2_3=x_2_3;
while   (iter<=epoch)
    shuffled=randperm(N);
        for j=1:N
          rand_idx=shuffled(1,j);
          y_2_3(rand_idx,1)= sign((Weight_mat(rand_idx,:)*x_2_3)+bias);  %%update the state for this neuron
          aux=Weight_mat*y_2_3;
          energy_0(:,iter)=(-1/2*aux'*p0)-(bias*sum(y_2_3));
          energy_1(:,iter)=(-1/2 *aux'*p1)-(bias*sum(y_2_3));
          energy_2(:,iter)=(-1/2 *aux'*p2)-(bias*sum(y_2_3));

          m_0(:,iter)=1/N *(y_2_3'*p0);
          m_1(:,iter)=1/N *(y_2_3'*p1);
          m_2(:,iter)=1/N *(y_2_3'*p2);
        end
        
    if(isequal(y_2_3,x_2_3)) %dovrebbe arrestarsi in poche iterazioni
		break;
	end
        iter=iter+1;  
        x_2_3=y_2_3;
end

acc_2_3 = (1- 1/N*sum(y_2_3~=p2))*100;
figure;
subplot(2,1,1)
imagesc(reshape(d_2_3,32,32));
title('Distorted Image, d-2-3')
hold on
subplot(2,1,2)
imagesc(reshape(y_2_3,32,32))
title(sprintf('%s %.2f %s','reconstructed image,',acc_2_3,'%'))
% saveas(gcf,'distorted_2_3_reconstructed.png')

figure;
plot (1:iter, energy_0)
hold on
plot (1:iter, energy_1)
hold on
plot (1:iter, energy_2)
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Energy-0','Energy-1','Energy-2')
title('Energy Functions Related to d-2-3')
% saveas(gcf,'distorted_2_3_energy.png')

figure;
plot (1:1:iter, m_0)
hold on
plot (1:1:iter, m_1)
hold on
plot (1:1:iter, m_2)
set(gca,'YLim',[0 1])
xbounds = [1 iter];
set(gca,'XTick',xbounds(1):xbounds(2));
xlabel('epoch')
legend('Overlap-0','Overlap-1','Overlap-2')
title('Overlap Functions Related to d-2-3')
% saveas(gcf,'distorted_2_3_overlap.png')