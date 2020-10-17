function Bout = learn_B_new(Y, Wg, Bpre, XF, nu)

% Y: m * c
% Wg = L * c
% XF: m * L
% Bpre: m * L

% B = Bpre;
B = zeros(size(Bpre));
% B = sign(XF);

Q = nu * XF + Y*Wg'; 
L = size(B,2);

for time = 1:10
    Z0 = B;
    for k = 1:L %closed form for each row of B
        Zk = B;
        Zk(:,k) = []; %ignore bit k
        Wkk = Wg(k,:);
        Wk = Wg;
        Wk(k,:) = [];
        B(:,k) = sign( Q(:,k) - Zk*Wk*Wkk' );
    end
%     if norm(B-Z0,'fro') < 1e-10 * norm (Z0,'fro')
%         break;
%     end
end

Bout = B;


