% cnn_train_test.m

% === 1. Set Direktori Dataset ===
trainFolder = fullfile('dataset', 'train');
testFolder  = fullfile('dataset', 'test');

% === 2. ImageDatastore untuk otomatis baca dan label ===
imdsTrain = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsTest = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% === 3. Resize semua gambar ke ukuran tetap ===
inputSize = [100 100];  % ukuran gambar
imdsTrain.ReadFcn = @(filename)imresize(imread(filename), inputSize);
imdsTest.ReadFcn = @(filename)imresize(imread(filename), inputSize);

% === 4. Definisikan Arsitektur CNN ===
layers = [
    imageInputLayer([100 100 3])

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

% === 5. Opsi Pelatihan ===
options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize',16, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsTest, ...
    'Verbose',false, ...
    'Plots','training-progress');

% === 6. Latih CNN ===
net = trainNetwork(imdsTrain, layers, options);

% === 7. Evaluasi Model ===
YPred = classify(net, imdsTest);
YTrue = imdsTest.Labels;

accuracy = sum(YPred == YTrue) / numel(YTrue);
confMat = confusionmat(YTrue, YPred);

disp(['✅ Akurasi: ', num2str(accuracy*100, '%.2f'), '%']);

% === 8. Simpan hasil ke .mat ===
save('cnn_model.mat', 'net', 'accuracy', 'confMat', 'YPred', 'YTrue');
