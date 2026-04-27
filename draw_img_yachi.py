#!/usr/bin/python3
# coding:UTF-8
# ピッタリ収まるように拡大・縮小
# グラデーションによるグレースケール化に対応
# USAGE: ./draw_img.py 画像横サイズ 画像縦サイズ データファイル名
## ##箇所は萢が追記

import numpy as np
import math
import numpy as np
import os
import cv2

img_x = 150  # 画像サイズ：横
img_y = 150  # 画像サイズ：縦

model_name = "prott5"  ## モデル名
method_name = "pca_cog"  ## 使った次元削減の手法
dir_name = "fold_pca_cog"

maxpix = 255  # グレースケール最大階調値


# 参照、移動の方向
hear, right, left, up, down, up_right = \
    [np.array(list) for list in [[0, 0], [1, 0],
                                 [-1, 0], [0, 1], [0, -1], [1, 1]]]

def drawline(from_p,  dest_p, pixv):  # from_p-dest_pを結ぶ直線を画素値 pixv で描画する

    dx, dy = dest_p-from_p

    if dx < 0:  # 終点が第2，3象限にある場合は逆向きに描画
        dest_p, from_p = from_p, dest_p
        dx, dy = dest_p-from_p

    n_vec = np.array([dy, -dx])  # 直線の法線ベクトル

    start = np.array([int(np.floor(p)) for p in from_p])  # 移動の出発点
    endp = np.array([int(np.floor(p)) for p in dest_p])  # 移動の終点

    pix[tuple(start)] = pixv # 出発点の画素値を設定

    movp = np.copy(start)  # 移動点の初期化
    # 直線が right と up_right の間を通っているとき right に移動、そうでなければmov2に移動
    # mov2：直線の終点が第1象限にあるとき up、 第4象限にあるとき down
    mov2 = up if dy >= 0 else down

    while not (np.allclose(movp, endp)):  # 移動点が終点に達するまで繰り返し

        vec1 = movp + right - from_p
        vec2 = movp + up_right - from_p

        if np.dot(n_vec, vec1)*np.dot(n_vec, vec2) <= 0: # right と up_right の間を通過
            movp += right
        else:
            movp += mov2
        pix[tuple(movp)] = pixv  # 移動後の画素値を設定

for fold in range(1, 6):
    print("Loading...")
    train_vecs = f"./data/embedding-vectors_exclusion/{model_name}/{dir_name}/fold{fold}_train_{method_name}"  # 埋め込みベクトルの場所
    train_files = [os.path.join(train_vecs, f) for f in os.listdir(train_vecs) if f.endswith(".npy")]  # .npy ファイルを全て取得
    train_output_dir = f"./graphs/{model_name}_exclusion/{dir_name}_cv2/fold{fold}_train_{method_name}_ut"  # グラフ表示画像の保存先
    os.makedirs(train_output_dir, exist_ok=True)

    test_vecs = f"./data/embedding-vectors_exclusion/{model_name}/{dir_name}/fold{fold}_test_{method_name}"  # 埋め込みベクトルの場所
    test_files = [os.path.join(test_vecs, f) for f in os.listdir(test_vecs) if f.endswith(".npy")]  # .npy ファイルを全て取得
    test_output_dir = f"./graphs/{model_name}_exclusion/{dir_name}_cv2/fold{fold}_test_{method_name}_ut"  # グラフ表示画像の保存先
    os.makedirs(test_output_dir, exist_ok=True)

    print(f"fold{fold}:", end="")
    for filename in train_files:
        ## 配列の初期化
        img = np.array([float(img_x), float(img_y)])
        pix = np.zeros((img_x, img_y), dtype=int)  # 画素値（ゼロで初期化）

        # データファイルの読み込み
        dat = np.load(filename, mmap_mode="r")
        # print(dat)
        dat = np.cumsum(dat, axis=0)
        base = os.path.basename(filename).replace(".npy", ".png")
        save_filename = os.path.join(train_output_dir, base)

        # 拡大・縮小
        max, min = np.max(dat, axis=0), np.min(dat, axis=0)  # 各軸に沿った最大と最小
        width, height = max-min  # 画像のオリジナルサイズ
        imgw, imgh = img-1  # 描画領域の横・縦の最大値

        rat = np.array([width/imgw, height/imgh]) # 縦横それぞれピッタリ収める場合
        # rat = np.max([width/imgw, height/imgh]) # 縦横の大きい方をピッタリ収める場合
        dat = dat/rat  # 1/ratにリスケール
        max, min = (max,  min)/rat

        # 中心が一致するように平行移動
        mid = (max+min)/2.0
        dat = [row-mid+img/2. for row in dat]

        # 描画
        for i in range(len(dat)-1):
            drawline(dat[i], dat[i+1], math.floor((maxpix+1)/len(dat)*i) ) # グレースケール
            # drawline(dat[i], dat[i+1], 1 ) # 2値画像
        
        ## pngで保存
        cv2.imwrite(save_filename, np.flipud(pix.T))

    for filename in test_files:
        ## 配列の初期化
        img = np.array([float(img_x), float(img_y)])
        pix = np.zeros((img_x, img_y), dtype=int)  # 画素値（ゼロで初期化）

        # データファイルの読み込み
        dat = np.load(filename, mmap_mode="r")
        # print(dat)
        dat = np.cumsum(dat, axis=0)
        base = os.path.basename(filename).replace(".npy", ".png")
        save_filename = os.path.join(test_output_dir, base)

        # 拡大・縮小
        max, min = np.max(dat, axis=0), np.min(dat, axis=0)  # 各軸に沿った最大と最小
        width, height = max-min  # 画像のオリジナルサイズ
        imgw, imgh = img-1  # 描画領域の横・縦の最大値

        rat = np.array([width/imgw, height/imgh]) # 縦横それぞれピッタリ収める場合
        #rat = np.max([width/imgw, height/imgh]) # 縦横の大きい方をピッタリ収める場合
        dat = dat/rat  # 1/ratにリスケール
        max, min = (max,  min)/rat

        # 中心が一致するように平行移動
        mid = (max+min)/2.0
        dat = [row-mid+img/2. for row in dat]

        # 描画
        for i in range(len(dat)-1):
            drawline(dat[i], dat[i+1], math.floor((maxpix+1)/len(dat)*i) ) # グレースケール
            # drawline(dat[i], dat[i+1], 1 ) # 2値画像
        
        ## pngで保存
        cv2.imwrite(save_filename, np.flipud(pix.T))
    print("done")
