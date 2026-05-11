/*
 * ミスあり元のコード　もう使わない
 */




/*
 * Runの時に必要な動作
 * 1. DAPI画像のフォルダを選択
 * 2. MHC画像のフォルダを選択
 * 3. 自動処理開始
 */

macro "Count DAPI and Merge MHC" {
    // ディレクトリの選択
    dirDAPI = getDirectory("Choose DAPI Directory");
    dirMHC = getDirectory("Choose MHC Directory");
    parentDir = File.getParent(dirDAPI);
        
    // ファイルリストの取得
    listDAPI = getFileList(dirDAPI);
    listMHC = getFileList(dirMHC);
    
    // ファイル数チェック
    if (listDAPI.length != listMHC.length) {
        showMessage("DAPIとMHCのファイル数が一致しません。");
    }  

    // 高速化のために画像を作るの禁止（Batch Mode）
    setBatchMode(true);
    
    // --- 結果データ保存用のリスト準備 ---
    nameList = newArray(0);
    scoreList = newArray(0);
    dapiList = newArray(0);
    mhcList = newArray(0);
    

    // ---------------- ループ処理開始 ----------------
    for (i = 0; i < listDAPI.length; i++) {
        // 余計なものは読み込まない
        if (endsWith(listDAPI[i], ".tif")) {
            
            // --- Step 1: DAPIの処理  ---
            
            // Open DAPI image
            open(dirDAPI + listDAPI[i]);
            dapiID = getImageID();
            dapiTitle = getTitle(); // 元の名前を保存
            
            // DAPIの処理
            run("8-bit");
			run("Despeckle");
			run("Auto Threshold", "method=Mean white");
			run("Watershed");
			run("Analyze Particles...", "size=0.01-Infinity show=Nothing display clear");

            countDAPI = nResults; // 結果の行数＝粒子数

            // --- Step 2: MHCの処理  ---
            
            // Open MHC image
            open(dirMHC + listMHC[i]);
            mhcID = getImageID();
            mhcTitle = getTitle(); 
            
            // MHCの処理
            run("Enhance Contrast", "saturated=0.35");
            setMinAndMax(0, 85);
            run("8-bit");
            run("Despeckle");
            run("Auto Threshold", "method=Mean white");

            
            // --- Step 3: AND演算とカウント ---
            
            // Image Calculator -> AND
            imageCalculator("AND create", dapiTitle, mhcTitle);
            resultID = getImageID();	
            
            // 合成画像確認用
//            if (i == 1) { 
//	            saveAs("Tiff", parentDir + "Merged_" + dapiTitle);
//	            // 保存するとタイトルが変わることがあるのでID再取得（念の為）
//	            resultID = getImageID();
//            }
                    
            // Analyze Particles 
            run("Analyze Particles...", "size=0.01-Infinity show=Nothing display clear");
            merge_count = nResults;	//解析結果の行数 = 粒子の数
            
            
            // --- Step 4:　成熟度スコア算出・表の準備 ---
            
            if (countDAPI > 0) {
			    maturity_score = merge_count / countDAPI;
			} else {
			    maturity_score = 0;
			}
			
			// print(i + " 番目の成熟度は" + maturity_score);
			
			// リストに追加
            nameList = Array.concat(nameList, dapiTitle);
            scoreList = Array.concat(scoreList, maturity_score);
            dapiList = Array.concat(dapiList, countDAPI);
            mhcList = Array.concat(mhcList, merge_count);
            

            // --- 開いた画像を閉じる ---
            selectImage(resultID); close();
            selectImage(mhcID); close();
            selectImage(dapiID); close();
            
            // 解析に使った標準Resultsテーブル削除
            run("Clear Results");

        }   //if文終わり   
    }	//for文終わり


	// ---------------- ループ終了・出力 ----------------

	summaryTableName = "Annalysis Table";
    Table.create(summaryTableName);
    
    for (j = 0; j < nameList.length; j++) {	
    	//Table.set("列名",行数,数値);
        Table.set("Image Name", j, nameList[j]);
        Table.set("Maturity score", j, scoreList[j]);
        Table.set("DAPI Count", j, dapiList[j]);
        Table.set("MHC Count", j, mhcList[j]);
    }
    
    Table.update();  //Table.setで埋めたテーブルをアップデートして表示させる

    // 保存処理
    
    savePath = parentDir + File.separator + "Analysis_Result.csv";
    
    saveAs("Results", savePath);
    
    setBatchMode(false);
    showMessage("Done!", "完了しました！\n\n保存先:\n" + savePath);

