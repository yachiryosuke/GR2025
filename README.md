# GR2025

## ファイルの実行順

1. 埋め込みベクトルの取得
  - `prott5-xl-get-embedding.ipynb`
2. 5分割データの作成
  - `fasta-copy-kfold.ipynb` 
3. 重心の位置ベクトル群の算出
  - `prott5-center-of-gravity-kfold.ipynb`
4. PCAを用いた1段階目の次元圧縮
  - `dimensionality-reduction-pca-cog-kfold.ipynb`
5. t-SNEを用いた2段階目の次元圧縮
  - `dimensionality-reduction-pcaopentsne-cog-kfold.ipynb`
6. 2次元ベクトルのスケーリング
  - `scale-the-embedding-10-170-kfold.ipynb`
7. グラフ表示画像の作成
  - `draw_img_yachi.py`
8. 層化5分割交差検証
  - `cnn_prott5.ipynb`
