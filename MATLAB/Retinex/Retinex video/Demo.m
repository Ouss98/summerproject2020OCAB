clear; % Clear all variables from the workspace

[FileName,PathName] = uigetfile({'Video files (*.mpg;*.mp4);*.mpg;*mp4'},'Pick a video');% Select a file
readerobj = VideoReader([PathName,FileName]); % Use the pathname and file name to read the video

rate = readerobj.FrameRate; % Get the frame rate
h = readerobj.Height; % Get the height
w = readerobj.Width; % Get the width

readerobj.CurrentTime = 0; % Specify that reading should start at 0 seconds from the beginning.
        

% Read in all video frames.
index = 1;  % Initialise a sequence of frames with zeros. 
while hasFrame(readerobj) % hasFrame -> determine if a video is still available to read
        vidFrame = readFrame(readerobj); 
        video(:,:,:,index) = vidFrame; % There are four dimensions because this is a colour video
        index = index + 1; % Increment the frame index
end
u_video = uint8(video);
number_of_frames = size(video,4);
%******************************************
implay(video,rate); % play original video
scene_frames(:,:,:,1) = video(:,:,:,1); 
%for each frame apply Retinex_Var function
for i = 1:number_of_frames
    frame = video(:,:,:,i);
%     denoisedFrame = convn(double(frame), ones(3)/9, 'same');
%     denoisedFrame = uint8(denoisedFrame); % Gaussian noise removal
%     enhanced(:,:,:,i) = Retinex_Var(denoisedFrame);
% ********************************************************************
% Deblurring
    hf = fft2(h,size(frame,1),size(frame,2));
    sigma_u = 10^(-40/20)*abs(1-0);
    cam_blur = real(ifft2(hf.*fft2(frame)));
    cam_noise = cam_blur + sigma_u*randn(size(cam_blur));
    cam_pinv = real(ifft2((abs(hf) > 0.1).*fft2(cam_noise)./hf));
    enhanced(:,:,:,i) = Retinex_Var(cam_pinv);
% ********************************************************************
    %enhanced(:,:,:,i) = Retinex_Var(frame);
end
u_enhanced = uint8(enhanced);
implay(u_enhanced,rate); % play enhanced video
