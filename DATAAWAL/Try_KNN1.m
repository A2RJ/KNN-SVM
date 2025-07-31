clc, clear all;
file = '/Users/siskaatmawanoktavia/Downloads/THESIS EDITING /APP DIABETIC RETINOPATHY/data siska.xlsx';
x_train = readtable(file,'Sheet', 'Train','Range','B1:E159');
y_train = readtable(file,'Sheet', 'Train','Range','F1:F159');
x_test = readtable(file,'Sheet', 'Test', 'Range','B1:E60');
y_test = readtable(file,'Sheet', 'Test', 'Range','F1:F60');
% nfiles = length(x_train);    % Number of files found
% fprintf ("x_train : %i",nfiles)
% 
% nfiles = length(img_files);    % Number of files found
% fprintf ("jumlah data : %i",nfiles)

modelknn = fitcknn(x_train,y_train,'NumNeighbors',2,'Standardize',1);
modelsvm = fitcsvm(x_train,y_train);
saveLearnerForCoder(modelknn,'KNN_Classification_data_siska');
saveLearnerForCoder(modelsvm,'SVM_Classification_data_siska');

dtknn_benar = 0;
dtknn_salah = 0;
dtsvm_benar = 0;
dtsvm_salah = 0;

for i=1:height(y_test)
    label = predict(modelknn,x_test(i,[1 2 3 4]));
    predict_knn = label;
    label = predict(modelsvm,x_test(i,[1 2 3 4]));
    predict_svm = label;
    
    if string(predict_knn) == string(y_test{i,1})
        dtknn_benar = dtknn_benar +1;
    else
        dtknn_salah = dtknn_salah +1;
    end
    
    if string(predict_svm) == string(y_test{i,1})
        dtsvm_benar = dtsvm_benar +1;
    else
        dtsvm_salah = dtsvm_salah +1;
    end
end

KNN_acc_testing = dtknn_benar*100/height(y_test);
SVM_acc_testing = dtsvm_benar*100/height(y_test);

% fprintf ("Jumlah prediksi KNN benar = %i", dtknn_benar);
% fprintf ("\nJumlah prediksi KNN salah = %i", dtknn_salah);
fprintf ("\nPersentase akurasi testing KNN = %i", KNN_acc_testing);


% fprintf ("\n\nJumlah prediksi SVM benar = %i", dtsvm_benar);
% fprintf ("\nJumlah prediksi SVM salah = %i", dtsvm_salah);
fprintf ("\nPersentase akurasi testing SVM = %i", SVM_acc_testing);

dtknn_benar = 0;
dtknn_salah = 0;
dtsvm_benar = 0;
dtsvm_salah = 0;

for i=1:height(y_train)
    label = predict(modelknn,x_train(i,[1 2 3 4]));
    predict_knn = label;
    label = predict(modelsvm,x_train(i,[1 2 3 4]));
    predict_svm = label;
    
    if string(predict_knn) == string(y_train{i,1})
        dtknn_benar = dtknn_benar +1;
    else
        dtknn_salah = dtknn_salah +1;
    end
    
    if string(predict_svm) == string(y_train{i,1})
        dtsvm_benar = dtsvm_benar +1;
    else
        dtsvm_salah = dtsvm_salah +1;
    end
end

KNN_acc_training = dtknn_benar*100/height(y_train);
SVM_acc_training = dtsvm_benar*100/height(y_train);

fprintf ("\nPersentase akurasi training KNN = %i", KNN_acc_training);
fprintf ("\nPersentase akurasi training SVM = %i", SVM_acc_training);

KNN_acc = (KNN_acc_training + KNN_acc_testing)/2;
SVM_acc = (SVM_acc_training + SVM_acc_testing)/2;

fprintf ("\nPersentase akurasi KNN = %i", KNN_acc);
fprintf ("\nPersentase akurasi SVM = %i", SVM_acc);


load_model = '/Users/siskaatmawanoktavia/Downloads/THESIS EDITING /APP DIABETIC RETINOPATHY/KNN_Classification_data_siska.mat';
model = loadLearnerForCoder(load_model);

disp(class(x_train(i,[1 2 3 4])));
label = predict(model,x_train(i,[1 2 3 4]));
% disp (label)
