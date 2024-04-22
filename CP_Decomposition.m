% CP Decomposition example.
% David Grenier: david_grenier@uri.edu
% CP Decomposition based on algorithm described in Kolda and Bader (2009).
% Creates a toy 2x2x2 tensor and performs a rank two CP Decomposition.
%
% Note, this code is written to be readable and illustrative, not
% efficient.


% Set seed, N, error_norms, threshold
clear; clf;
rng(1,"twister");
N =  100000;        % max iteration
error_norms = [];   % list for holding error norms
threshold = 1e-8;   % stopping threshold

% Create random 2x2x2 tensor
X = randi(5, 2, 2, 2);

% Create mode-n unfoldings
X1 = [X(:,:,1), X(:,:,2)];
X2 = [X(:,:,1)', X(:,:,2)'];
X3 = [
    X(:,1,1)', X(:,2,1)';
    X(:,1,2)', X(:,2,2)';
    ];

% Initialialize A, B, C
A = randi(5, 2);
B = randi(5, 2);
C = randi(5, 2);

% Begin loop
for n = 1:N
    % Update A, B, C
    VA = times(B'*B, C'*C);
    A = X1 * [kron(C(:,1), B(:,1)), kron(C(:,2), B(:,2))] * pinv(VA);
    LambdaA = [norm(A(:,1)), norm(A(:,2))];
    A = [A(:,1)/norm(A(:,1)), A(:,2)/norm(A(:,2))];

    VB = times(A'*A, C'*C);
    B = X2 * [kron(C(:,1), A(:,1)), kron(C(:,2), A(:,2))] * pinv(VB);
    LambdaB = [norm(B(:,1)), norm(B(:,2))];
    B = [B(:,1)/norm(B(:,1)), B(:,2)/norm(B(:,2))];

    VC = times(A'*A, B'*B);
    C = X3 * [kron(B(:,1), A(:,1)), kron(B(:,2), A(:,2))] * pinv(VC);
    LambdaC = [norm(C(:,1)), norm(C(:,2))];
    C = [C(:,1)/norm(C(:,1)), C(:,2)/norm(C(:,2))];

    % Approximate X with Y (Y = hat X)
    Y = zeros(2,2,2);
    for r = 1:2
        for i = 1:2
            Y(:,:,i) = Y(:,:,i) + LambdaA(r) * A(:,r) * B(:,r)' * C(i,r);
        end
    end

    % Get approximation error and test improvement of fit for stopping
    % criteria
    error = X - Y;
    error_norm = norm(error(:));
    error_norms = [error_norms, error_norm];
    if n > 1  
        if abs(error_norms(end-1) - error_norms(end)) < threshold
            break
        end
    end
end

plot(error_norms(2:end));