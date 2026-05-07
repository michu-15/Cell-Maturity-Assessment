macro "Count DAPI and Merge MHC" {

    dirDAPI = getDirectory("Choose DAPI Directory");
    dirMHC  = getDirectory("Choose MHC Directory");
    parentDir = File.getParent(dirDAPI);

    rawDAPI = getFileList(dirDAPI);

    // DAPIファイルだけ抽出
    listDAPI = newArray(0);
    keyList  = newArray(0);

    for (i = 0; i < rawDAPI.length; i++) {
        name = rawDAPI[i];
        lower = toLowerCase(name);

        if (endsWith(lower, "_dapi.tif")) {
            listDAPI = Array.concat(listDAPI, name);
            keyList  = Array.concat(keyList, getMiddleNumber(name));
        }
    }

    sortByKey(listDAPI, keyList);

    setBatchMode(true);
    run("Clear Results");

    // 結果保存用
    nameList      = newArray(0);
    dapiFileList  = newArray(0);
    mhcFileList   = newArray(0);
    scoreList     = newArray(0);
    dapiList      = newArray(0);
    mergedList    = newArray(0);

    for (i = 0; i < listDAPI.length; i++) {

        dapiFile = listDAPI[i];

        // MHCファイル名生成
        mhcFile = replace(dapiFile, "_DAPI.tif", "_MHC.tif");

        if (!File.exists(dirMHC + mhcFile)) {
            print("対応するMHCファイルが見つかりません: " + mhcFile);
            continue;
        }

        baseName = replace(dapiFile, "_DAPI.tif", "");

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
        nameList     = Array.concat(nameList, baseName);
        dapiFileList = Array.concat(dapiFileList, dapiFile);
        mhcFileList  = Array.concat(mhcFileList, mhcFile);
        scoreList    = Array.concat(scoreList, maturity_score);
        dapiList     = Array.concat(dapiList, countDAPI);
        mergedList   = Array.concat(mergedList, merge_count);

        // 閉じる
        selectImage(resultID); close();
        selectImage(mhcID); close();
        selectImage(dapiID); close();
    }

    // デバッグ用
    print("DAPI候補数: " + listDAPI.length);
    for (i = 0; i < listDAPI.length; i++) print(listDAPI[i]);

    // ---------------- 出力 ----------------
    summaryTableName = "Analysis Table";
    Table.create(summaryTableName);

    for (i = 0; i < nameList.length; i++) {
        Table.set("Image Name", i, nameList[i]);
        Table.set("DAPI File Used", i, dapiFileList[i]);
        Table.set("MHC File Used", i, mhcFileList[i]);
        Table.set("Maturity score", i, scoreList[i]);
        Table.set("DAPI Count", i, dapiList[i]);
        Table.set("Merged Count", i, mergedList[i]);
    }

    Table.update();

    savePath = parentDir + File.separator + "Count_Result.csv";
    Table.save(savePath);

    setBatchMode(false);
    showMessage("Done!", "完了しました！\n\n保存先:\n" + savePath);
}


// ---------------- 補助関数 ----------------

// No.X_YYYYY_DAPI.tif → YYYYY抽出
function getMiddleNumber(filename) {
    s = replace(filename, "_DAPI.tif", "");

    uscore = lastIndexOf(s, "_");
    if (uscore < 0) return -1;

    numStr = substring(s, uscore + 1, lengthOf(s));
    return parseInt(numStr);
}


// ソート
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
            tempKey = keyArray[i];
            keyArray[i] = keyArray[minIndex];
            keyArray[minIndex] = tempKey;

            tempName = nameArray[i];
            nameArray[i] = nameArray[minIndex];
            nameArray[minIndex] = tempName;
        }
    }
}