function GWKernel = GaborWavelet (width, height, kMax, f, u , v, delta)
% Create the Gabor Kernel
% Modified of Chai Zhi's  28-12-2013
delta2 = (delta)^2;
kv = kMax / (f^v);
thetaU = (u*pi)/10;
kuv = kv * exp (i * thetaU);
kuv2 = abs(kuv)^2;
GWKernel = zeros ( height , width );
for y =  -height/2+1 : height/2
    
    for x = -width/2+1 : width/2
        GWKernel(y+height/2,x+width/2) = (kuv2 / delta2 ) * exp( -0.5 * kuv2 * ( x ^ 2 + y ^ 2 ) / delta2) * ( exp( i * ( real( kuv ) * y + imag ( kuv ) * x ) ) - exp ( -0.5 * delta2 ) );
    end
    
end
