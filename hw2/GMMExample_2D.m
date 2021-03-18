
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
R1 = chol(sigma1); %對sigma1進行Cholesky分解
X1 = randn(m1, 2) * R1; %產生常態分佈的亂數
X1 = X1 + repmat(mu1, size(X1, 1), 1);  %將X1加上mean

R2 = chol(sigma2);
X2 = randn(m2, 2) * R2;
X2 = X2 + repmat(mu2, size(X2, 1), 1);

X = [X1; X2];

%%=====================================================
%% STEP 1b: Plot the data points and their pdfs.

figure(1); 

% Display a scatter plot of the two distributions.
hold off;   %當新圖出現時，取消原圖
plot(X1(:, 1), X1(:, 2), 'bo'); %畫上X1資料點
hold on;    %使當前軸以及圖形保持不被重新整理，已達多圖共存
plot(X2(:, 1), X2(:, 2), 'ro'); %畫上X2資料點

set(gcf,'color','white') % White background for the figure.

% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
gridSize = 100;
u = linspace(-6, 6, gridSize);  %在-6和6之間分成100個點
[A B] = meshgrid(u, u); %將u轉換成對應的網絡座標
gridX = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.
z1 = gaussianND(gridX, mu1, sigma1);  %計算pdf
z2 = gaussianND(gridX, mu2, sigma2);

% Reshape the responses back into a 2D grid to be plotted with contour.
Z1 = reshape(z1, gridSize, gridSize);   %將剛剛算出的pdf轉到2D的網格上
Z2 = reshape(z2, gridSize, gridSize);

% Plot the contour lines to show the pdf over the data.
[C, h] = contour(u, u, Z1); %劃出pdf的等高線圖
[C, h] = contour(u, u, Z2);

axis([-6 6 -6 6])   %設置圖軸的範圍
title('Original Data and PDFs');    %設置圖的標題

%set(h,'ShowText','on','TextStep',get(h,'LevelStep')*2);


%%====================================================
%% STEP 2: Choose initial values for the parameters.

% Set 'm' to the number of data points.
m = size(X, 1);

k = 2;  % The number of clusters.
n = 2;  % The vector lengths.

% Randomly select k data points to serve as the initial means.
indeces = randperm(m);
mu = X(indeces(1:k), :);    %隨機挑選k筆資料當作初始的mean

sigma = [];

% Use the overal covariance of the dataset as the initial variance for each cluster.
for (j = 1 : k)
    sigma{j} = cov(X);  %使用dataset的overall covarience當作每類初始的varience
end

% Assign equal prior probabilities to each cluster.
phi = ones(1, k) * (1 / k); %每類給定各類相同的事前機率

%%===================================================
%% STEP 3: Run Expectation Maximization

% Matrix to hold the probability that each data point belongs to each cluster.
% One row per data point, one column per cluster.
W = zeros(m, k);    %矩陣W放的是每一點屬於那一類的機率為多少

% Loop until convergence.
for (iter = 1:1000) %iteration直到收斂
    
    fprintf('  EM Iteration %d\n', iter);   %印出目前的iteraion次數

    %%===============================================
    %% STEP 3a: Expectation
    %
    % Calculate the probability for each data point for each distribution.
    
    % Matrix to hold the pdf value for each every data point for every cluster.
    % One row per data point, one column per cluster.
    pdf = zeros(m, k);  %pdf為每一點帶進各類機率分布函數的值
    
    % For each cluster...
    for (j = 1 : k)
        
        % Evaluate the Gaussian for all data points for cluster 'j'.
        pdf(:, j) = gaussianND(X, mu(j, :), sigma{j}); %將每一點對應的mean和sigma帶入機率分布函數
    end
    
    % Multiply each pdf value by the prior probability for cluster.
    %    pdf  [m  x  k]
    %    phi  [1  x  k]   
    %  pdf_w  [m  x  k]
    pdf_w = bsxfun(@times, pdf, phi);   %將pdf算出的值在乘上事前機率，得到當前blob的機率值 α_b*P(x|μ_b,V_b)
    
    % Divide the weighted probabilities by the sum of weighted probabilities for each cluster.
    %   sum(pdf_w, 2) -- sum over the clusters.
    W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2)); %將pdf_w除上每個群算出來機率值的和 α_b*P(x|μ_b,V_b)/?i=1~k α_i P(x│μ_i,V_i) 
    
    %%===============================================
    %% STEP 3b: Maximization
    %%
    %% Calculate the probability for each data point for each distribution.

    % Store the previous means.
    prevMu = mu;    %儲存上次的mean
    
    % For each of the clusters...
    for (j = 1 : k)
    
        % Calculate the prior probability for cluster 'j'.
        phi(j) = mean(W(:, j), 1);  %更新事前機率 αnew=1/N * ?i=1~k P(b|x_i,μ_b,V_b)
        
        % Calculate the new mean for cluster 'j' by taking the weighted
        % average of all data points.
        mu(j, :) = weightedAverage(W(:, j), X); %更新mean μ_new=?i=1~k x_i*P(b|x_i,μ_b,V_b)/?_(i=1~k P(b|x_i,μ_b,V_b)

        % Calculate the covariance matrix for cluster 'j' by taking the 
        % weighted average of the covariance for each training example. 
        
        %147-158更新covarience matrix
        sigma_k = zeros(n, n); %assign 0 到covarience matrix
        
        % Subtract the cluster mean from all data points.
        Xm = bsxfun(@minus, X, mu(j, :));   %將資料減去mean
        
        % Calculate the contribution of each training example to the covariance matrix.
        for (i = 1 : m)
            sigma_k = sigma_k + (W(i, j) .* (Xm(i, :)' * Xm(i, :)));  % ?i=1~k P(b│x_i,μ_b,V_b)(x_i-μ_new)'(x_i-μ_new )   
        end
        
        % Divide by the sum of weights.
        sigma{j} = sigma_k ./ sum(W(:, j)); %更新covarience matrix ?i=1~k P(b│x_i,μ_b,V_b)(x_i-μ_new)'(x_i-μ_new ) / ?i=1~k P(b|x_i,μ_b,V_b)
    end
    
    % Check for convergence.
    if (mu == prevMu)   %更新的mean等於前一個mean=>表示已收斂
        break
    end
            
% End of Expectation Maximization    
end

%%=====================================================
%% STEP 4: Plot the data points and their estimated pdfs.

% Display a scatter plot of the two distributions.
figure(2);  
hold off; %當新圖出現時，取消原圖
plot(X1(:, 1), X1(:, 2), 'bo'); %畫上X1資料點
hold on; %使當前軸以及圖形保持不被重新整理，已達多圖共存
plot(X2(:, 1), X2(:, 2), 'ro');  %畫上X2資料點

set(gcf,'color','white') % White background for the figure.

plot(mu1(1), mu1(2), 'kx'); %在mean值也就是中心點，標上叉叉
plot(mu2(1), mu2(2), 'kx');

% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
gridSize = 100; 
u = linspace(-6, 6, gridSize); %在-6和6之間分成100個點
[A B] = meshgrid(u, u); %將u轉換成對應的網絡座標
gridX = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.
z1 = gaussianND(gridX, mu(1, :), sigma{1}); %把最終得到的mean和covarience matrix代入計算pdf
z2 = gaussianND(gridX, mu(2, :), sigma{2});

% Reshape the responses back into a 2D grid to be plotted with contour.
Z1 = reshape(z1, gridSize, gridSize); %將算出的pdf轉到2D的網格上
Z2 = reshape(z2, gridSize, gridSize);

% Plot the contour lines to show the pdf over the data.
[C, h] = contour(u, u, Z1); %劃出pdf的等高線圖
[C, h] = contour(u, u, Z2);
axis([-6 6 -6 6])  %設置圖軸的範圍

title('Original Data and Estimated PDFs'); %設置圖的標題