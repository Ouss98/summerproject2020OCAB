faceDetector = vision.CascadeObjectDetector('FrontalFaceCART'); % Create a detector object.
img = imread('people.jpg'); % Read input image.
img = rgb2gray(img); % Convert RGB image to gray.
bbox = step(faceDetector,img); % Detect faces.
outputImg = insertObjectAnnotation(img, 'rectangle', bbox, 'Face'); % Annotate detected faces.

figure(1);
subplot(1,2,1);imshow(img);title('Input Image');
subplot(1,2,2);imshow(outputImg);title('Detected Faces');

% Count the faces and show the result in a msgbox
[m,n] =  size(int32(bbox));
uiwait(msgbox(sprintf('Face Count = %d',m),'Detected Faces','modal'));

% Draw a yellow rectangle around the detected faces.
hold on
for i = 1:size(bbox,1)
    rectangle('position',bbox(i,:),'Linewidth',2,'Linestyle','-','Edgecolor','y');
end

hold on
N = size(bbox,1);
handles.N = N;
counter = 1;
for i = 1:N
    face = imcrop(img,bbox(i,:)); % Crop the detected faces.
    savenam = strcat('./' ,num2str(counter), '.jpg'); 
    baseDir  = './TestDatabase/';
    newName  = [baseDir num2str(counter) '.jpg']; 
    handles.face = face;
    while exist(newName,'file')
        counter = counter + 1;
        newName = [baseDir num2str(counter) '.jpg'];
    end
    faceResized = imresize(face,[112,92]); % Resize the faces.
    imwrite(faceResized,newName); % Save the cropped faces.
    
    figure(2);
    imshow(face); 
    title('Cropped face');
    pause(.5);
end