% cnn_train_test_complete.m
% =====================================================================
% COMPLETE CNN IMPLEMENTATION FOR IMAGE CLASSIFICATION
% Improved version with better architecture and evaluation
% =====================================================================

clc; clear; close all;

%% === 1. SETUP DATASET DIRECTORIES ===
fprintf('🔧 Setting up dataset directories...\n');
trainFolder = fullfile('dataset', 'train');
testFolder  = fullfile('dataset', 'test');

% Check if directories exist
if ~exist(trainFolder, 'dir')
    error('❌ Training folder not found: %s', trainFolder);
end
if ~exist(testFolder, 'dir')
    error('❌ Testing folder not found: %s', testFolder);
end

%% === 2. CREATE IMAGE DATASTORES ===
fprintf('📁 Creating image datastores...\n');
imdsTrain = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsTest = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Display dataset info
fprintf('📊 Dataset Information:\n');
fprintf('   Training images: %d\n', numel(imdsTrain.Files));
fprintf('   Testing images: %d\n', numel(imdsTest.Files));
fprintf('   Classes: %s\n', strjoin(string(categories(imdsTrain.Labels)), ', '));

% Count samples per class
labelCounts = countEachLabel(imdsTrain);
disp(labelCounts);

%% === 3. IMAGE PREPROCESSING ===
fprintf('🖼️ Setting up image preprocessing...\n');
inputSize = [100 100];  % Target image size

% Custom read function with preprocessing
imdsTrain.ReadFcn = @(filename)preprocessImage(filename, inputSize);
imdsTest.ReadFcn = @(filename)preprocessImage(filename, inputSize);

%% === 4. DATA AUGMENTATION (Optional) ===
fprintf('🔄 Setting up data augmentation...\n');
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20, 20], ...
    'RandXTranslation', [-10, 10], ...
    'RandYTranslation', [-10, 10], ...
    'RandXScale', [0.9, 1.1], ...
    'RandYScale', [0.9, 1.1], ...
    'RandXReflection', true);

% Apply augmentation to training data
augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augImdsTest = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

%% === 5. IMPROVED CNN ARCHITECTURE ===
fprintf('🧠 Building CNN architecture...\n');
numClasses = numel(categories(imdsTrain.Labels));

layers = [
    % Input Layer
    imageInputLayer([100 100 3], 'Name', 'input', 'Normalization', 'zerocenter')
    
    % First Convolutional Block
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    % Second Convolutional Block
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    
    % Third Convolutional Block
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
    
    % Fourth Convolutional Block (Optional - for more complex datasets)
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    globalAveragePooling2dLayer('Name', 'gap')
    
    % Fully Connected Layers
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.5, 'Name', 'dropout1')
    
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu_fc2')
    dropoutLayer(0.3, 'Name', 'dropout2')
    
    % Output Layer
    fullyConnectedLayer(numClasses, 'Name', 'fc_output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];

% Analyze network
analyzeNetwork(layers);

%% === 6. TRAINING OPTIONS ===
fprintf('⚙️ Setting up training options...\n');
options = trainingOptions('adam', ...
    'MaxEpochs', 25, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 8, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augImdsTest, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% === 7. TRAIN THE CNN MODEL ===
fprintf('🚀 Starting CNN training...\n');
tic;
net = trainNetwork(augImdsTrain, layers, options);
trainingTime = toc;
fprintf('✅ Training completed in %.2f minutes\n', trainingTime/60);

%% === 8. MODEL EVALUATION ===
fprintf('📈 Evaluating model performance...\n');

% Predict on test set
[YPred, scores] = classify(net, augImdsTest);
YTrue = imdsTest.Labels;

% Calculate overall accuracy
accuracy = sum(YPred == YTrue) / numel(YTrue);
fprintf('🎯 Overall Accuracy: %.2f%%\n', accuracy*100);

% Confusion Matrix
confMat = confusionmat(YTrue, YPred);
classes = categories(YTrue);

% Display confusion matrix
fprintf('\n📊 Confusion Matrix:\n');
fprintf('True\\Pred\t');
for i = 1:length(classes)
    fprintf('%s\t', char(classes{i}));
end
fprintf('\n');
for i = 1:length(classes)
    fprintf('%s\t\t', char(classes{i}));
    for j = 1:length(classes)
        fprintf('%d\t', confMat(i,j));
    end
    fprintf('\n');
end

% Per-class metrics
fprintf('\n📋 Per-Class Performance:\n');
fprintf('Class\t\tPrecision\tRecall\t\tF1-Score\n');
fprintf('-----\t\t---------\t------\t\t--------\n');

for i = 1:length(classes)
    tp = confMat(i,i);
    fp = sum(confMat(:,i)) - tp;
    fn = sum(confMat(i,:)) - tp;
    
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1score = 2 * (precision * recall) / (precision + recall);
    
    fprintf('%s\t\t%.3f\t\t%.3f\t\t%.3f\n', ...
        char(classes{i}), precision, recall, f1score);
end

%% === 9. VISUALIZATION ===
fprintf('📊 Creating visualizations...\n');

% Confusion Matrix Chart
figure('Position', [100, 100, 500, 400]);
confusionchart(YTrue, YPred);
title('Confusion Matrix');
saveas(gcf, 'confusion_matrix.png');

% ROC Curve (for binary classification)
if length(classes) == 2
    figure('Position', [650, 100, 500, 400]);
    [X, Y, T, AUC] = perfcurve(YTrue, scores(:,2), classes{2});
    plot(X, Y, 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(sprintf('ROC Curve (AUC = %.3f)', AUC));
    grid on;
    saveas(gcf, 'roc_curve.png');
    fprintf('🎯 AUC Score: %.3f\n', AUC);
end

% Sample predictions visualization
figure('Position', [100, 600, 800, 600]);
numSamples = min(16, numel(imdsTest.Files));
subplot_rows = ceil(sqrt(numSamples));
subplot_cols = ceil(numSamples / subplot_rows);

for i = 1:numSamples
    subplot(subplot_rows, subplot_cols, i);
    
    % Read and display image
    img = readimage(imdsTest, i);
    imshow(img);
    
    % Get prediction confidence
    pred_class = char(YPred(i));
    true_class = char(YTrue(i));
    confidence = max(scores(i,:)) * 100;
    
    % Color code: green for correct, red for incorrect
    if YPred(i) == YTrue(i)
        title_color = [0, 0.7, 0]; % Green
        status = '✓';
    else
        title_color = [0.8, 0, 0]; % Red
        status = '✗';
    end
    
    title(sprintf('%s %s\nPred: %s (%.1f%%)\nTrue: %s', ...
        status, pred_class, pred_class, confidence, true_class), ...
        'Color', title_color, 'FontSize', 8);
    axis off;
end
sgtitle('Sample Predictions', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'sample_predictions.png');

%% === 10. SAVE RESULTS ===
fprintf('💾 Saving results...\n');

% Create results structure
results = struct();
results.net = net;
results.accuracy = accuracy;
results.confusionMatrix = confMat;
results.predictions = YPred;
results.trueLabels = YTrue;
results.scores = scores;
results.classes = classes;
results.trainingTime = trainingTime;
results.inputSize = inputSize;

% Save to .mat file
save('cnn_model.mat', 'results', '-v7.3');

% Save detailed report
reportFile = 'cnn_training_report.txt';
fid = fopen(reportFile, 'w');
fprintf(fid, '=== CNN TRAINING REPORT ===\n');
fprintf(fid, 'Date: %s\n', datestr(now));
fprintf(fid, 'Training Time: %.2f minutes\n', trainingTime/60);
fprintf(fid, 'Overall Accuracy: %.2f%%\n', accuracy*100);
fprintf(fid, 'Number of Classes: %d\n', length(classes));
fprintf(fid, 'Training Samples: %d\n', numel(imdsTrain.Files));
fprintf(fid, 'Testing Samples: %d\n', numel(imdsTest.Files));
fprintf(fid, '\nConfusion Matrix:\n');
for i = 1:length(classes)
    fprintf(fid, '%s: ', char(classes{i}));
    for j = 1:length(classes)
        fprintf(fid, '%d ', confMat(i,j));
    end
    fprintf(fid, '\n');
end
fclose(fid);

fprintf('✅ All results saved successfully!\n');
fprintf('📁 Files created:\n');
fprintf('   - cnn_model_complete.mat (model and results)\n');
fprintf('   - cnn_training_report.txt (detailed report)\n');
fprintf('   - confusion_matrix.png (confusion matrix chart)\n');
if length(classes) == 2
    fprintf('   - roc_curve.png (ROC curve)\n');
end
fprintf('   - sample_predictions.png (prediction samples)\n');

%% === HELPER FUNCTIONS ===
function img = preprocessImage(filename, targetSize)
    % Custom preprocessing function
    img = imread(filename);
    
    % Convert to RGB if grayscale
    if size(img, 3) == 1
        img = repmat(img, [1, 1, 3]);
    end
    
    % Resize
    img = imresize(img, targetSize);
    
    % Optional: Histogram equalization for better contrast
    % img = histeq(rgb2gray(img));
    % img = repmat(img, [1, 1, 3]);
end