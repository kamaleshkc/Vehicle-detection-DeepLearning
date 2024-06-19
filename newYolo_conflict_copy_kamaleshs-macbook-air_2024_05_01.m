annotations = readtable('BITVehicle/VehicleInfo.csv'); % Replace 'path/to/annotations.csv' with the actual path to your CSV file

 % Replace 'path/to/images' with the directory containing your images
imds = imageDatastore("BITVehicle/");

% Step 3: Parse bounding box annotations
imageFileNames = annotations.name;

% Step 3: Preprocess bounding box annotations
numImages = size(annotations, 1);

bboxes = cell(numImages,1);

for i = 1:numImages
    % Extract bounding box coordinates
    top = annotations.top(i);
    left = annotations.left(i);
    right = annotations.right(i);
    bottom = annotations.bottom(i);

    % Convert to [x, y, width, height] format
    x = left;
    y = top;
    width = right - left;
    height = bottom - top;
    % Store bounding box in [x, y, width, height] format
    bboxes{i} = [x, y, width, height];
    
end
    Vehicle=vertcat(bboxes);
    dataTable = table(Vehicle);
    blds = boxLabelDatastore(dataTable(:,1:end));


ds = combine(imds, blds);

net = load('yolov2VehicleDetector.mat');
lgraph = net.lgraph;

lgraph.Layers;


options = trainingOptions('sgdm',...
          'InitialLearnRate',0.001,...
          'Verbose',true,...
          'MiniBatchSize',16,...
          'MaxEpochs',30,...
          'Shuffle','never',...
          'VerboseFrequency',30,...
          'CheckpointPath',tempdir);

[detector,info] = trainYOLOv2ObjectDetector(ds,lgraph,options);

plot(info.TrainingLoss)
grid on
xlabel('Number of Iterations')
ylabel('Training Loss for Each Iteration')

%% % Create a VideoReader object to read the video
video = VideoReader('testvideo.mp4');

% Create a VideoWriter object to write the output video with detections
outputVideo = VideoWriter('output_video.mp4', 'MPEG-4');

% Open the VideoWriter object
open(outputVideo);

% Loop through each frame of the video
while hasFrame(video)
    % Read the current frame
    frame = readFrame(video);

    % Run the trained YOLO v2 object detector on the current frame for vehicle detection
    [bboxes, scores] = detect(detector, frame);

    % Display the detection results if any bounding boxes are found
    if ~isempty(bboxes)
        % Scale down the bounding boxes
        scaling_factor = 0.5; % You can adjust this factor as needed
        bboxes_scaled = bboxes * scaling_factor;

        % Adjust the positions of scaled bounding boxes
        bboxes_scaled(:, 1:2) = bboxes(:, 1:2) * scaling_factor;

        % Insert scaled bounding boxes with scores into the frame
        frame = insertObjectAnnotation(frame, 'rectangle', bboxes_scaled, scores);
    end

    % Write the frame with detection results to the output video
    writeVideo(outputVideo, frame);
    step(videoPlayer, outputVideo);
end

% Close the VideoWriter object
close(outputVideo);
%% 
implay('output_video.mp4');

