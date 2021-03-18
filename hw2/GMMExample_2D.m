
%%======================================================
%% STEP 1a: Generate data from two 2D distributions.

mu1 = [1 2];       % Mean
sigma1 = [ 3 .2;   % Covariance matrix
          .2  2];
m1 = 200;          % Number of data points.

mu2 = [-1 -2];
sigma2 = [2 0;
          0 1];
m2 = 100;

% Generate sample points with the specified means and covariance matrices.
R1 = chol(sigma1); %��sigma1�i��Cholesky����
X1 = randn(m1, 2) * R1; %���ͱ`�A���G���ü�
X1 = X1 + repmat(mu1, size(X1, 1), 1);  %�NX1�[�Wmean

R2 = chol(sigma2);
X2 = randn(m2, 2) * R2;
X2 = X2 + repmat(mu2, size(X2, 1), 1);

X = [X1; X2];

%%=====================================================
%% STEP 1b: Plot the data points and their pdfs.

figure(1); 

% Display a scatter plot of the two distributions.
hold off;   %��s�ϥX�{�ɡA�������
plot(X1(:, 1), X1(:, 2), 'bo'); %�e�WX1����I
hold on;    %�Ϸ�e�b�H�ιϧΫO�����Q���s��z�A�w�F�h�Ϧ@�s
plot(X2(:, 1), X2(:, 2), 'ro'); %�e�WX2����I

set(gcf,'color','white') % White background for the figure.

% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
gridSize = 100;
u = linspace(-6, 6, gridSize);  %�b-6�M6��������100���I
[A B] = meshgrid(u, u); %�Nu�ഫ�������������y��
gridX = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.
z1 = gaussianND(gridX, mu1, sigma1);  %�p��pdf
z2 = gaussianND(gridX, mu2, sigma2);

% Reshape the responses back into a 2D grid to be plotted with contour.
Z1 = reshape(z1, gridSize, gridSize);   %�N����X��pdf���2D������W
Z2 = reshape(z2, gridSize, gridSize);

% Plot the contour lines to show the pdf over the data.
[C, h] = contour(u, u, Z1); %���Xpdf�������u��
[C, h] = contour(u, u, Z2);

axis([-6 6 -6 6])   %�]�m�϶b���d��
title('Original Data and PDFs');    %�]�m�Ϫ����D

%set(h,'ShowText','on','TextStep',get(h,'LevelStep')*2);


%%====================================================
%% STEP 2: Choose initial values for the parameters.

% Set 'm' to the number of data points.
m = size(X, 1);

k = 2;  % The number of clusters.
n = 2;  % The vector lengths.

% Randomly select k data points to serve as the initial means.
indeces = randperm(m);
mu = X(indeces(1:k), :);    %�H���D��k����Ʒ�@��l��mean

sigma = [];

% Use the overal covariance of the dataset as the initial variance for each cluster.
for (j = 1 : k)
    sigma{j} = cov(X);  %�ϥ�dataset��overall covarience��@�C����l��varience
end

% Assign equal prior probabilities to each cluster.
phi = ones(1, k) * (1 / k); %�C�����w�U���ۦP���ƫe���v

%%===================================================
%% STEP 3: Run Expectation Maximization

% Matrix to hold the probability that each data point belongs to each cluster.
% One row per data point, one column per cluster.
W = zeros(m, k);    %�x�}W�񪺬O�C�@�I�ݩ󨺤@�������v���h��

% Loop until convergence.
for (iter = 1:1000) %iteration���즬��
    
    fprintf('  EM Iteration %d\n', iter);   %�L�X�ثe��iteraion����

    %%===============================================
    %% STEP 3a: Expectation
    %
    % Calculate the probability for each data point for each distribution.
    
    % Matrix to hold the pdf value for each every data point for every cluster.
    % One row per data point, one column per cluster.
    pdf = zeros(m, k);  %pdf���C�@�I�a�i�U�����v������ƪ���
    
    % For each cluster...
    for (j = 1 : k)
        
        % Evaluate the Gaussian for all data points for cluster 'j'.
        pdf(:, j) = gaussianND(X, mu(j, :), sigma{j}); %�N�C�@�I������mean�Msigma�a�J���v�������
    end
    
    % Multiply each pdf value by the prior probability for cluster.
    %    pdf  [m  x  k]
    %    phi  [1  x  k]   
    %  pdf_w  [m  x  k]
    pdf_w = bsxfun(@times, pdf, phi);   %�Npdf��X���Ȧb���W�ƫe���v�A�o���eblob�����v�� �\_b*P(x|�g_b,V_b)
    
    % Divide the weighted probabilities by the sum of weighted probabilities for each cluster.
    %   sum(pdf_w, 2) -- sum over the clusters.
    W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2)); %�Npdf_w���W�C�Ӹs��X�Ӿ��v�Ȫ��M �\_b*P(x|�g_b,V_b)/?i=1~k �\_i P(x�x�g_i,V_i) 
    
    %%===============================================
    %% STEP 3b: Maximization
    %%
    %% Calculate the probability for each data point for each distribution.

    % Store the previous means.
    prevMu = mu;    %�x�s�W����mean
    
    % For each of the clusters...
    for (j = 1 : k)
    
        % Calculate the prior probability for cluster 'j'.
        phi(j) = mean(W(:, j), 1);  %��s�ƫe���v �\new=1/N * ?i=1~k P(b|x_i,�g_b,V_b)
        
        % Calculate the new mean for cluster 'j' by taking the weighted
        % average of all data points.
        mu(j, :) = weightedAverage(W(:, j), X); %��smean �g_new=?i=1~k x_i*P(b|x_i,�g_b,V_b)/?_(i=1~k P(b|x_i,�g_b,V_b)

        % Calculate the covariance matrix for cluster 'j' by taking the 
        % weighted average of the covariance for each training example. 
        
        %147-158��scovarience matrix
        sigma_k = zeros(n, n); %assign 0 ��covarience matrix
        
        % Subtract the cluster mean from all data points.
        Xm = bsxfun(@minus, X, mu(j, :));   %�N��ƴ�hmean
        
        % Calculate the contribution of each training example to the covariance matrix.
        for (i = 1 : m)
            sigma_k = sigma_k + (W(i, j) .* (Xm(i, :)' * Xm(i, :)));  % ?i=1~k P(b�xx_i,�g_b,V_b)(x_i-�g_new)'(x_i-�g_new )   
        end
        
        % Divide by the sum of weights.
        sigma{j} = sigma_k ./ sum(W(:, j)); %��scovarience matrix ?i=1~k P(b�xx_i,�g_b,V_b)(x_i-�g_new)'(x_i-�g_new ) / ?i=1~k P(b|x_i,�g_b,V_b)
    end
    
    % Check for convergence.
    if (mu == prevMu)   %��s��mean����e�@��mean=>��ܤw����
        break
    end
            
% End of Expectation Maximization    
end

%%=====================================================
%% STEP 4: Plot the data points and their estimated pdfs.

% Display a scatter plot of the two distributions.
figure(2);  
hold off; %��s�ϥX�{�ɡA�������
plot(X1(:, 1), X1(:, 2), 'bo'); %�e�WX1����I
hold on; %�Ϸ�e�b�H�ιϧΫO�����Q���s��z�A�w�F�h�Ϧ@�s
plot(X2(:, 1), X2(:, 2), 'ro');  %�e�WX2����I

set(gcf,'color','white') % White background for the figure.

plot(mu1(1), mu1(2), 'kx'); %�bmean�Ȥ]�N�O�����I�A�ФW�e�e
plot(mu2(1), mu2(2), 'kx');

% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
gridSize = 100; 
u = linspace(-6, 6, gridSize); %�b-6�M6��������100���I
[A B] = meshgrid(u, u); %�Nu�ഫ�������������y��
gridX = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.
z1 = gaussianND(gridX, mu(1, :), sigma{1}); %��̲ױo�쪺mean�Mcovarience matrix�N�J�p��pdf
z2 = gaussianND(gridX, mu(2, :), sigma{2});

% Reshape the responses back into a 2D grid to be plotted with contour.
Z1 = reshape(z1, gridSize, gridSize); %�N��X��pdf���2D������W
Z2 = reshape(z2, gridSize, gridSize);

% Plot the contour lines to show the pdf over the data.
[C, h] = contour(u, u, Z1); %���Xpdf�������u��
[C, h] = contour(u, u, Z2);
axis([-6 6 -6 6])  %�]�m�϶b���d��

title('Original Data and Estimated PDFs'); %�]�m�Ϫ����D