% Tucker1 Decomposition example.
% David Grenier: david_grenier@uri.edu
% Tucker1 based on algorithm described in Kolda and Bader (2009).
% Creates a toy 2x2x2 tensor and performs a rank two Tucker1 Decomposition.
%
% Note, this code is written to be readable and illustrative, not
% efficient.

clear; clf;

% Set seed for reproducibility
rng(1, 'twister');

% Create X
X = randi(5,2,2,2);

% Create mode-n unfoldings
X1 = [X(:,:,1), X(:,:,2)];
X2 = [X(:,:,1)', X(:,:,2)'];
X3 = [
    X(:,1,1)', X(:,2,1)';
    X(:,1,2)', X(:,2,2)';
    ];

% Get A, B, C
[A, S, V] = svds(X1, 2);
[B, S, V] = svds(X2, 2);
[C, S, V] = svds(X3, 3);

% Get G
G = zeros(2,2,2); % initialize
G1 = A'*X1; % get G1
G(:,1,1) = G1(:,1); % refold
G(:,2,1) = G1(:,2); % refold
G(:,1,2) = G1(:,3); % refold
G(:,2,2) = G1(:,4); % refold

G2 = [G(:,:,1)', G(:,:,2)'];
G2 = B'*G2;
G(1,:,1) = G2(:,1); % refold
G(2,:,1) = G2(:,2); % refold
G(1,:,2) = G2(:,3); % refold
G(2,:,2) = G2(:,4); % refold

G3 = [
    G(:,1,1)', G(:,2,1)';
    G(:,1,2)', G(:,2,2)';
    ];
G3 = C'*G3;
G(:,1,1) = G3(1,1:2); % refold
G(:,2,1) = G3(1,3:4); % refold
G(:,1,2) = G3(2,1:2); % refold
G(:,2,2) = G3(2,3:4); % refold

% reconstruct and test
Xhat = zeros(2,2,2);
for i = 1:2
    for j = 1:2
        for k = 1:2
            for p = 1:2
                for q = 1:2
                    for r = 1:2
                        Xhat(i,j,k) = Xhat(i,j,k) + (G(p,q,r) * A(i,p) * B(j,q) * C(k,r));
                    end
                end
            end
        end
    end
end

error = X - Xhat;
error_norm = norm(error(:));
