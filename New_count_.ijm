macro "Count DAPI and Merge MHC" {

    dirDAPI = getDirectory("Choose DAPI Directory");
    dirMHC  = getDirectory("Choose MHC Directory");
    parentDir = File.getParent(dirDAPI);

    rawDAPI = getFileList(dirDAPI);

    // DAPIファイルだけ抽出
    listDAPI = newArray(0);
    keyList  = newArray(0);   // ソート用の数値キー(XXXXX)

    for (i = 0; i < rawDAPI.length; i++) {
        name = rawDAPI[i];
        lower = toLowerCase(name);

        if (endsWith(lower, "_dapi.tif") || endsWith(lower, "_dapi.tiff")) {
            listDAPI = Array.concat(listDAPI, name);
            keyList  = Array.concat(keyList, getMiddleNumber(name));
        }
    }

    // XXXXX の数値でソート
    sortByKey(listDAPI, keyList);

    setBatchMode(true);
    run("Clear Results");

    // 結果保存用
    nameList  = newArray(0);
    scoreList = newArray(0);
    dapiList  = newArray(0);
    mhcList   = newArray(0);

    for (i = 0; i < listDAPI.length; i++) {

        dapiFile = listDAPI[i];

        // 対応するMHCファイル名を直接生成
        mhcFile = replace(dapiFile, "_DAPI.tif", "_MHC.tif");


        // MHCファイル存在確認
        if (!File.exists(dirMHC + mhcFile)) {
            print("対応するMHCファイルが見つかりません: " + mhcFile);
            continue;
        }

        // 共通名
        baseName = dapiFile;
        baseName = replace(baseName, "_DAPI.tif", "");

        // ---------------- DAPI処理 ----------------
        open(dirDAPI + dapiFile);
        dapiID = getImageID();
        dapiTitle = getTitle();

        run("8-bit");
        run("Despeckle");
        run("Auto Threshold", "method=Mean white");
        run("Watershed");
        run("Analyze Particles...", "size=0.01-Infinity show=Nothing display clear");

        countDAPI = nResults;
        run("Clear Results");

        // ---------------- MHC処理 ----------------
        open(dirMHC + mhcFile);
        mhcID = getImageID();
        mhcTitle = getTitle();

        run("Enhance Contrast", "saturated=0.35");
        setMinAndMax(0, 85);
        run("8-bit");
        run("Despeckle");
        run("Auto Threshold", "method=Mean white");

        // ---------------- AND演算 ----------------
        imageCalculator("AND create", dapiTitle, mhcTitle);
        resultID = getImageID();

        run("Analyze Particles...", "size=0.01-Infinity show=Nothing display clear");
        merge_count = nResults;
        run("Clear Results");

        // ---------------- スコア計算 ----------------
        if (countDAPI > 0) {
            maturity_score = merge_count / countDAPI;
        } else {
            maturity_score = 0;
        }

        // 結果保存
        nameList  = Array.concat(nameList, baseName);
        scoreList = Array.concat(scoreList, maturity_score);
        dapiList  = Array.concat(dapiList, countDAPI);
        mhcList   = Array.concat(mhcList, merge_count);

        // 閉じる
        selectImage(resultID); close();
        selectImage(mhcID); close();
        selectImage(dapiID); close();
    }

    // ---------------- 出力 ----------------
    summaryTableName = "Analysis Table";
    Table.create(summaryTableName);

    for (i = 0; i < nameList.length; i++) {
        Table.set("Image Name", i, nameList[i]);
        Table.set("Maturity score", i, scoreList[i]);
        Table.set("DAPI Count", i, dapiList[i]);
        Table.set("MHC Count", i, mhcList[i]);
    }

    Table.update();

    savePath = parentDir + File.separator + "Analysis_Result.csv";
    Table.save(savePath);

    setBatchMode(false);
    showMessage("Done!", "完了しました！\n\n保存先:\n" + savePath);
}


// ---------------- 補助関数 ----------------

// ファイル名 No.X_XXXXX_DAPI.tif から XXXXX を数値で取り出す
function getMiddleNumber(filename) {
    s = filename;

    s = replace(s, "_DAPI.tif", "");

    uscore = indexOf(s, "_");
    if (uscore < 0) return -1;

    numStr = substring(s, uscore + 1, lengthOf(s));
    return parseInt(numStr);
}


// keyList の数値に従って listDAPI を昇順ソート
function sortByKey(nameArray, keyArray) {
    n = lengthOf(keyArray);

    for (i = 0; i < n - 1; i++) {
        minIndex = i;

        for (j = i + 1; j < n; j++) {
            if (keyArray[j] < keyArray[minIndex]) {
                minIndex = j;
            }
        }

        if (minIndex != i) {
            // key入れ替え
            tempKey = keyArray[i];
            keyArray[i] = keyArray[minIndex];
            keyArray[minIndex] = tempKey;

            // name入れ替え
            tempName = nameArray[i];
            nameArray[i] = nameArray[minIndex];
            nameArray[minIndex] = tempName;
        }
    }
}