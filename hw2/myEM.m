function [W mu sigma]=myEM(X,k)
    %% Choose initial values for the parameters.
%     [row,col]=size(img);
%     % X=double(reshape(img,row*col/2,2));
%     X=double(img);
%     N=size(X,1); %The number of data points.
%     k=3; % The number of clusters.
%     n=col; % The vector lengths.
    [N,dim]=size(X);

    indeces=randperm(N);
    mu=X(randi([1,N],1,k),:); %隨機挑選k筆資料當作初始的mean

    sigma=[];
    for j=1:k
        sigma{j}=cov(X);  %使用dataset的overall covarience當作每類初始的varience
    end

    phi=ones(1,k)*(1/k); %每類給定各類相同的事前機率

    %% Run Expectation Maximization
    W=zeros(N,k);

    for iter=1:1000
        fprintf('  EM Iteration %d\n', iter);
          %% a.Expectation
        pdf=zeros(N,k);
        for j=1:k
            pdf(:,j)=gaussianND(X,mu(j,:),sigma{j}); %將每一點對應的mean和sigma帶入機率分布函數
        end
        pdf_w = bsxfun(@times, pdf, phi); %將pdf算出的值在乘上事前機率，得到當前blob的機率值 α_b*P(x|μ_b,V_b)
        W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2)); %將pdf_w除上每個群算出來機率值的和 α_b*P(x|μ_b,V_b)/?i=1~k α_i P(x│μ_i,V_i) 
           %% b.Maximization
        prevMu = mu; %儲存上次的mean
        for j=1:k
            phi(j) = mean(W(:, j), 1);  %更新事前機率 αnew=1/N * sum_i=1~k P(b|x_i,μ_b,V_b)
            mu(j,:)=weightedAverage(W(:,j),X); %更新mean μ_new=sum_i=1~k x_i*P(b|x_i,μ_b,V_b)/sum_i=1~k P(b|x_i,μ_b,V_b)
            %更新covarience matrix
            sigma_k = zeros(dim, dim); %assign 0 到covarience matrix
            Xm = bsxfun(@minus, X, mu(j, :));   %將資料減去mean        
            % Calculate the contribution of each training example to the covariance matrix.
            for (i = 1 : N)
                sigma_k = sigma_k + (W(i, j) .* (Xm(i, :)' * Xm(i, :)));  % sum_i=1~k P(b│x_i,μ_b,V_b)(x_i-μ_new)'(x_i-μ_new )   
            end        
            % Divide by the sum of weights.
            sigma{j} = sigma_k ./ sum(W(:, j)); %更新covarience matrix sum_i=1~k P(b│x_i,μ_b,V_b)(x_i-μ_new)'(x_i-μ_new ) / sum_i=1~k P(b|x_i,μ_b,V_b)
        end

        if (mu == prevMu)   %更新的mean等於前一個mean=>表示已收斂
            break
        end
    end
end
