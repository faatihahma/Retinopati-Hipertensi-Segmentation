function [N, CRAE, CRVE, AVR] = calAVR(Art,Vein)

% AVR
if size(Art,1)>=6 && size(Vein,1)>=6
    A = sortrows(Art,'descend'); 
    A(1,2) = 0.88*sqrt(A(1,1)^2+A(6,1)^2); A(2,2) = 0.88*sqrt(A(2,1)^2+A(5,1)^2); A(3,2) = 0.88*sqrt(A(3,1)^2+A(4,1)^2);
    A(1,3) = 0.88*sqrt(A(1,2)^2+A(3,2)^2); A(2,3) = A(2,2); 
    A(1,4) = 0.88*sqrt(A(1,3)^2+A(2,3)^2);

    V = sortrows(Vein,'descend');
    V(1,2) = 0.95*sqrt(V(1,1)^2+V(6,1)^2); V(2,2) = 0.95*sqrt(V(2,1)^2+V(5,1)^2); V(3,2) = 0.95*sqrt(V(3,1)^2+V(4,1)^2);
    V(1,3) = 0.95*sqrt(V(1,2)^2+V(3,2)^2); V(2,3) = V(2,2); 
    V(1,4) = 0.95*sqrt(V(1,3)^2+V(2,3)^2);

    AVR = A(1,4)/V(1,4); N = 6; CRAE = A(1,4); CRVE = V(1,4);
elseif size(Art,1)>=3 && size(Vein,1)>=3
    A = sortrows(Art,'descend'); 
    A(1,2) = 0.88*sqrt(A(1,1)^2+A(3,1)^2); A(2,2) = A(1,2);
    A(1,3) = 0.88*sqrt(A(1,2)^2+A(2,2)^2);

    V = sortrows(Vein,'descend');
    V(1,2) = 0.95*sqrt(V(1,1)^2+V(3,1)^2); V(2,2) = V(1,2);
    V(1,3) = 0.95*sqrt(V(1,2)^2+V(2,2)^2);

    AVR = A(1,3)/V(1,3); N = 3; CRAE = A(1,3); CRVE = V(1,3);
elseif size(Art,1)>=2 && size(Vein,1)>=2
    A = sortrows(Art,'descend'); 
    A(1,2) = 0.88*sqrt(A(1,1)^2+A(2,1)^2);

    V = sortrows(Vein,'descend');
    V(1,2) = 0.95*sqrt(V(1,1)^2+V(2,1)^2);

    AVR = A(1,2)/V(1,2); N = 2; CRAE = A(1,2); CRVE = V(1,2);
elseif size(Art,1)>=1 && size(Vein,1)>=1
    A = sortrows(Art,'descend'); 
    V = sortrows(Vein,'descend');
    AVR = A(1,1)*0.88/(V(1,1)*0.95); N = 1; CRAE = A(1,1)*0.88; CRVE = V(1,1)*0.95;
else
    AVR = 0; N = 0; CRAE = 0; CRVE = 0;
end

end

