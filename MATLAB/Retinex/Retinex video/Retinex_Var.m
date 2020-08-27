function enhanced = Retinex_Var(frame)
HSV_image = rgb2hsv(frame); %RGB -> HSV
value = HSV_image(:,:,3)*255; %HSV (Hue, Saturation and Value)

%***********************************************************************
%WB=ones(size(value))*(1-max(max(HSV_image(:,:,3)))); % initial weight
% H=fspecial('average',[3 3]);
% denominator=imfilter(HSV_image(:,:,3).^2,H);
% denominator=denominator-imfilter(HSV_image(:,:,3),H).^2;
% denominator=sqrt(abs(denominator));
% WA=1./denominator;
% W=WA.*WB;
% [V,Diagonal] = eig(W'*W);
%***********************************************************************

L = imgaussfilt(value,.2); % Illumination - Init with Gaussian low-pass filter
R = value./(L+0.0001); % Reflectance
L0 = mean2(L); % Average of illumination
%[m,n] = size(value); % Size of HSV Value 
%constant = 1;
%FFT_constant = fft2(constant,m,n);
derivative_x = [0 0 0;-1 1 0;0 0 0];
derivative_y = [0 0 0;0 1 0;0 -1 0];
alpha = 10; beta = 0.1; gamma = 0.001; a = 10; 
% Defaut params : alpha = 10; beta = 0.1; gamma = 0.001; a = 10; 
iter = 4; % 4 iterations 
Tau = 0.05;
%*******************************************************************

%**************************************************************
for i=1:iter
    R=R+Tau*(value./(L+0.000001)-(1+beta*imfilter(imfilter(R,derivative_x),derivative_y))); % Reflectance computation
    % R Threshold parameter    
    index=find(R>1);
    R(index)=1;
    index=find(R<0);
    R(index)=0;
%     % %***********************************************
    L=L+Tau*(gamma*L0+value./(R+0.000001)-(1+gamma+alpha*imfilter(imfilter(R,derivative_x),derivative_y)));
    
    % L Threshold parameter
    index=find(L>value);
    L(index)=value(index);
%     index=find(L<0);
%     L(index)=0;
end


L_adjusted = 255*(2*atan(a * L/255)/pi); % Shrinking
% %****************************
L_final = adapthisteq(uint8(round(L_adjusted))); % adaptive Histogram equalization (CLAHE)
SV = R.*double(L_final); % New SV
HSV_enhanced_image = HSV_image;
HSV_enhanced_image(:,:,3) = SV; % Apply Retinex approach to the frame
enhanced_RGB = hsv2rgb(HSV_enhanced_image); % HSV -> RGB


enhanced = uint8(round(enhanced_RGB)); % uint8 -> Convert fixed-point numeric object to unsigned 8-bit integer
