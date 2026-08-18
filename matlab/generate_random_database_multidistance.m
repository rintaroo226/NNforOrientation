% generate_random_database_multidistance.m
% generate_random_database.m と同一の手順で学習用データベースを生成するが、
% 各サンプルごとに距離もランダムにサンプリングする。
%
% eval_distance_sweep.py で、distance=10固定で学習したモデルが他の距離では
% 精度が大きく落ちることが確認された。bboxクロップ前処理(bboxcrop_dataset.py)
% で見かけのスケールは正規化できるが、学習データが常に同じ実効解像度
% (bbox≈427px)でしか生成されていないため、遠距離特有のガタガタしたエッジに
% NNが慣れていない(分布シフト)ことが主因と考えられる。ここでは学習データ
% 自体にも距離のばらつきを持たせることで、様々な実効解像度をNNに学習させる。
%
% generate_random_database.m は変更せず、この用途専用のスクリプトとして
% 独立に用意する。initBoxSim/renderBoxImage は無改変で再利用する。

addpath(fileparts(mfilename('fullpath')));  % initBoxSim, renderBoxImage

%% 設定
N            = 10000;       % サンプル数 (train と同一)
rng_seed     = 43;          % train(42)/test(999)/distance_sweep(777)とは異なるシード

cam_params.imageSize = 512;
cam_params.fov       = 25;
target_size          = [3, 2, 1];  % [幅, 高さ, 奥行] (m)  train と同一

% 距離の範囲: 箱の半対角線 sqrt(1.5^2+1^2+0.5^2)=1.87m が
% FOV(半角12.5度)からはみ出さない最小距離は 1.87/tan(12.5°) ≈ 8.4m。
% 安全側に9mを下限とする(distance=5,7でクリッピングが起きた反省)。
% 上限はdistance-sweep評価(最大55m)より少し広めの60mとする。
dist_min = 9;
dist_max = 60;

save_dir  = 'database_random_multidistance';
image_dir = fullfile(save_dir, 'images');

%% 出力先作成
[~, ~] = mkdir(save_dir);
[~, ~] = mkdir(image_dir);

%% SO(3) 上で一様なクォータニオンをサンプリング（Shoemake の方法）
rng(rng_seed);
u1 = rand(N, 1);
u2 = rand(N, 1);
u3 = rand(N, 1);
qw = sqrt(1 - u1) .* sin(2 * pi * u2);
qx = sqrt(1 - u1) .* cos(2 * pi * u2);
qy = sqrt(u1)     .* sin(2 * pi * u3);
qz = sqrt(u1)     .* cos(2 * pi * u3);

%% 距離を対数一様にサンプリング (近距離・遠距離を偏りなくカバーするため)
log_dist   = log(dist_min) + (log(dist_max) - log(dist_min)) * rand(N, 1);
distances  = exp(log_dist);

%% 初期化（1回だけ）
handles = initBoxSim(cam_params.imageSize, target_size);
pose.tx = 0;
pose.ty = 0;

%% レンダリングループ
csv_path = fullfile(save_dir, 'labels.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, 'image,qw,qx,qy,qz,distance\n');

bin_threshold = 5;  % これより明るい画素を前景(255)とみなす（train と同一）

fprintf('=== 距離ランダム化学習用データベース生成 (%d枚, distance in [%.1f, %.1f]) ===\n', ...
    N, dist_min, dist_max);
tic;

for i = 1:N
    q = [qw(i), qx(i), qy(i), qz(i)];
    [pose.pitch, pose.yaw, pose.roll] = quatToEulerZYX(q);
    pose.distance = distances(i);

    img = renderBoxImage(handles, pose, cam_params);
    bin_img = uint8((img > bin_threshold) * 255);

    rel_path = sprintf('images/sample_%06d.png', i);
    imwrite(bin_img, fullfile(save_dir, rel_path));

    fprintf(fid, '%s,%.8f,%.8f,%.8f,%.8f,%.4f\n', ...
        rel_path, q(1), q(2), q(3), q(4), distances(i));

    if mod(i, 1000) == 0
        fprintf('  %d/%d 完了 (%.1f秒)\n', i, N, toc);
    end
end

fclose(fid);
close(handles.fig);

fprintf('\n完了。%d枚 + ラベルを "%s/" に保存。(%.1f秒)\n', N, save_dir, toc);


function [pitch_deg, yaw_deg, roll_deg] = quatToEulerZYX(q)
% generate_random_database.m と同一の変換。
w = q(1); x = q(2); y = q(3); z = q(4);

r11 = 1 - 2 * (y^2 + z^2);
r21 = 2 * (x * y + w * z);
r31 = 2 * (x * z - w * y);
r32 = 2 * (y * z + w * x);
r33 = 1 - 2 * (x^2 + y^2);

yaw_deg   = atan2d(-r31, sqrt(r11^2 + r21^2));
pitch_deg = atan2d(r32, r33);
roll_deg  = atan2d(r21, r11);
end
