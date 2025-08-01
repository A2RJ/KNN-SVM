function [hasilKNN, hasilSVM] = load_model_predict(fitur)
    % Pastikan pyenv aktif dan model Python dapat dipanggil
    try
        if count(py.sys.path, 'py_in_matlab') == 0
            insert(py.sys.path, int32(0), 'py_in_matlab');
        end
    catch ME
        error("❌ Python environment belum diatur dengan benar: %s", ME.message);
    end

    % Konversi fitur MATLAB ke Python list of float
    if ~isa(fitur, 'double') || numel(fitur) ~= 4
        error("Fitur harus berupa vektor double dengan 4 elemen.");
    end
    pyFitur = py.list(fitur);

    try
        % Panggil fungsi Python
        hasil = py.load_model_predict.predict(pyFitur);

        % Ambil hasil prediksi
        hasilKNN = string(hasil{1});
        hasilSVM = string(hasil{2});
    catch ME
        error("❌ Gagal memanggil Python: %s", ME.message);
    end
end
