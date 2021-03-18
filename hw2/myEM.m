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
    mu=X(randi([1,N],1,k),:); %�H���D��k����Ʒ�@��l��mean

    sigma=[];
    for j=1:k
        sigma{j}=cov(X);  %�ϥ�dataset��overall covarience��@�C����l��varience
    end

    phi=ones(1,k)*(1/k); %�C�����w�U���ۦP���ƫe���v

    %% Run Expectation Maximization
    W=zeros(N,k);

    for iter=1:1000
        fprintf('  EM Iteration %d\n', iter);
          %% a.Expectation
        pdf=zeros(N,k);
        for j=1:k
            pdf(:,j)=gaussianND(X,mu(j,:),sigma{j}); %�N�C�@�I������mean�Msigma�a�J���v�������
        end
        pdf_w = bsxfun(@times, pdf, phi); %�Npdf��X���Ȧb���W�ƫe���v�A�o���eblob�����v�� �\_b*P(x|�g_b,V_b)
        W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2)); %�Npdf_w���W�C�Ӹs��X�Ӿ��v�Ȫ��M �\_b*P(x|�g_b,V_b)/?i=1~k �\_i P(x�x�g_i,V_i) 
           %% b.Maximization
        prevMu = mu; %�x�s�W����mean
        for j=1:k
            phi(j) = mean(W(:, j), 1);  %��s�ƫe���v �\new=1/N * sum_i=1~k P(b|x_i,�g_b,V_b)
            mu(j,:)=weightedAverage(W(:,j),X); %��smean �g_new=sum_i=1~k x_i*P(b|x_i,�g_b,V_b)/sum_i=1~k P(b|x_i,�g_b,V_b)
            %��scovarience matrix
            sigma_k = zeros(dim, dim); %assign 0 ��covarience matrix
            Xm = bsxfun(@minus, X, mu(j, :));   %�N��ƴ�hmean        
            % Calculate the contribution of each training example to the covariance matrix.
            for (i = 1 : N)
                sigma_k = sigma_k + (W(i, j) .* (Xm(i, :)' * Xm(i, :)));  % sum_i=1~k P(b�xx_i,�g_b,V_b)(x_i-�g_new)'(x_i-�g_new )   
            end        
            % Divide by the sum of weights.
            sigma{j} = sigma_k ./ sum(W(:, j)); %��scovarience matrix sum_i=1~k P(b�xx_i,�g_b,V_b)(x_i-�g_new)'(x_i-�g_new ) / sum_i=1~k P(b|x_i,�g_b,V_b)
        end

        if (mu == prevMu)   %��s��mean����e�@��mean=>��ܤw����
            break
        end
    end
end
